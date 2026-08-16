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

from chess_anti_engine.eval.lc0_control_trainer import LIVE_TRAINER_PIN
from chess_anti_engine.train import losses as losses_module
from chess_anti_engine.train.losses import compute_loss, normalize_value_blend_fracs
from chess_anti_engine.train.target_builder import (
    CategoricalTargetParams,
    rebuild_categorical_target_in_arrays,
)
from chess_anti_engine.train.targets import (
    categorical_target_value,
    normalize_categorical_blend_fracs,
)
from chess_anti_engine.train.value_blend_guard import (
    PRODUCTION_GAME_FRAC,
    CategoricalRebuildReadout,
    ValueBlendMisconfigured,
    assert_categorical_rebuild_is_inert,
    assert_no_silent_outcome_fallback,
    assert_pid_cannot_reassert_sf_wdl,
    value_blend_readout,
)
from chess_anti_engine.tune.trainable_metrics import _dynamic_sf_wdl_weight

BATCH = 8
POLICY = 32


def _batch(*, with_sf_wdl: bool, with_search_wdl: bool = True) -> dict[str, torch.Tensor]:
    """A minimal batch shaped like an lc0-derived row set.

    lc0 rows carry `search_wdl` and `has_search_wdl` and NO `sf_wdl` —
    `with_sf_wdl=True` adds the production-shaped columns so the same test can
    show the leak closing when every row IS labelled.

    ⚑ `with_search_wdl=False` is review F1's state: `compute_loss` falls the
    SEARCH component back to the same raw one-hot, and that half had no
    instrument at all until 2026-08-16.
    """
    torch.manual_seed(0)
    batch: dict[str, torch.Tensor] = {
        "x": torch.zeros(BATCH, 175, 8, 8),
        "policy_t": torch.full((BATCH, POLICY), 1.0 / POLICY),
        "wdl_t": torch.zeros(BATCH, dtype=torch.int64),
        "has_policy": torch.ones(BATCH),
    }
    if with_search_wdl:
  # Deliberately not a one-hot and not degenerate, so the search component is
  # distinguishable from the outcome component in the built target.
        batch["search_wdl"] = torch.tensor([[0.5, 0.3, 0.2]]).repeat(BATCH, 1)
        batch["has_search_wdl"] = torch.ones(BATCH)
    if with_sf_wdl:
        batch["sf_wdl"] = torch.tensor([[0.2, 0.5, 0.3]]).repeat(BATCH, 1)
        batch["has_sf_wdl"] = torch.ones(BATCH)
    return batch


def _readout_from_compute_loss(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    **kwargs: Any,
) -> Any:
    """The guard's readout, built from the REAL `compute_loss` return values.

    ⚑ Not from hand-written row counts. The finding this closes was that the
    readout's inputs did not exist: `compute_loss` returned no search-side
    count at all, so no correct caller could have been written. Building the
    readout here through the same keys the driver reads means a rename or a
    dropped key fails this test rather than silently zeroing a term.
    """
    result = compute_loss(outputs, batch, **kwargs)
    return value_blend_readout(
        sf_wdl_frac=float(kwargs.get("sf_wdl_frac", 0.0)),
        search_wdl_frac=float(kwargs.get("search_wdl_frac", 0.0)),
        sf_effective_rows=float(result["sf_wdl_effective_rows"].item()),
        search_effective_rows=float(result["search_wdl_effective_rows"].item()),
        batch_rows=float(result["batch_rows"].item()),
    )


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
    component vectors) and `total` is asserted here, before any share is
    returned, so a blend that loses mass reads as lost mass instead of as an
    outcome share.

    ⚑ THE ORDER IS THE POINT, AND IT IS A CORRECTION. PR #438's write-up
    credited the `sums to 1.0` line with killing the `sf_component ->
    zeros_like` mutant. It did not: the caller's `shares["outcome"] ==
    0.80` assertion fired first and the mass check was never reached. Nor is
    the mass check redundant with `residual < 1e-5` — dropping the SF
    component leaves the target exactly in the span of the remaining
    components (`0.30*one_hot + 0.20*search`), so the residual is ~0 while the
    mass is 0.5. Checking mass FIRST makes each assertion own a defect the
    other cannot see.
    """
    import numpy as np

    one_hot = np.eye(3)[int(batch["wdl_t"][0].item())]
    names = ["outcome"]
    columns = [one_hot]
    if "sf_wdl" in batch:
        names.append("sf")
        columns.append(batch["sf_wdl"][0].numpy().astype(np.float64))
    if "search_wdl" in batch:
        names.append("search")
        columns.append(batch["search_wdl"][0].numpy().astype(np.float64))
    basis = np.stack(columns, axis=1)
    total = float(target[0].sum().item())
    assert total == pytest.approx(1.0, abs=1e-5), (
        f"the value target lost mass: it sums to {total}, not 1.0. A component "
        "that vanished and a component that MOVED are indistinguishable through "
        "the shares alone, and the residual cannot see it either — the "
        "remaining components still span the result."
    )
    coeffs, *_ = np.linalg.lstsq(basis, target[0].numpy().astype(np.float64), rcond=None)
    shares = dict(zip(names, (float(c) for c in coeffs), strict=True))
    residual = float(np.abs(basis @ coeffs - target[0].numpy()).max())
    assert residual < 1e-5, f"target is not a combination of its components ({residual})"
    shares["total"] = total
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


def test_missing_search_wdl_silently_moves_the_search_share_onto_the_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ REVIEW F1, THE DEFECT ITSELF, through the production loss.

    This is the arm's OWN blend — `sf_wdl_frac: 0.0`, `search_wdl_frac: 0.70` —
    on rows that carry no `has_search_wdl`. The value target is 100% raw game
    outcome. Before 2026-08-16 all three guards passed here and the readout
    printed `outcome_borne_frac 0.3000`, because nothing measured the search
    half: `compute_loss` did not even return a count for it.
    """
    batch, outputs = _batch(with_sf_wdl=False, with_search_wdl=False), _outputs()
    kwargs = {"sf_wdl_frac": 0.0, "search_wdl_frac": 0.70}
    target = _captured_wdl_target(monkeypatch, batch, outputs, **kwargs)

    shares = _decompose(target, batch)
    assert shares["outcome"] == pytest.approx(1.0, abs=1e-5)
    assert shares["total"] == pytest.approx(1.0, abs=1e-5)

    readout = _readout_from_compute_loss(batch, outputs, **kwargs)
    assert readout.search_effective_rows == 0.0
    assert readout.leaked_from_sf == 0.0, "the SF half is genuinely clean here"
    assert readout.leaked_from_search == pytest.approx(0.70)
    assert readout.outcome_borne_frac == pytest.approx(1.0)
    with pytest.raises(ValueBlendMisconfigured, match="RAW GAME OUTCOME"):
        assert_no_silent_outcome_fallback(readout, context="review F1")


def test_the_readout_agrees_with_the_target_compute_loss_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The readout's outcome share IS the target's, on both label regimes.

    ⚑ The pairing is the test. A readout that reports 0.30 while the solved
    target reads 1.00 is the exact failure F1 filed, and only a test that
    computes BOTH from the same call can see it — either number alone looks
    ordinary.
    """
    kwargs = {"sf_wdl_frac": 0.0, "search_wdl_frac": 0.70}
    for with_search in (True, False):
        batch, outputs = _batch(with_sf_wdl=False, with_search_wdl=with_search), _outputs()
        target = _captured_wdl_target(monkeypatch, batch, outputs, **kwargs)
        readout = _readout_from_compute_loss(batch, outputs, **kwargs)
        assert _decompose(target, batch)["outcome"] == pytest.approx(
            readout.outcome_borne_frac, abs=1e-5,
        ), f"readout disagrees with the built target (search labels: {with_search})"


def test_the_dampened_sf_share_lands_on_the_outcome_and_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ REVIEW F10. `sf_effective = has_sf_wdl * keep`, and the `1 - keep`
    shortfall the `sf_search_dampen_*` knobs remove falls onto the SAME one-hot.

    Both knobs are 0.0 in production and in the control today, so a leak read
    off `has_sf_wdl.sum()` agrees with the realized one BY COINCIDENCE. The
    yaml marks them live-tunable, and this is the trainer's production warning.
    Here SF says STM losing and search says STM winning on every row, so
    `sf_search_dampen_sf_low: 1.0` removes the whole SF component.
    """
    batch, outputs = _batch(with_sf_wdl=True), _outputs()
    batch["sf_wdl"] = torch.tensor([[0.1, 0.2, 0.7]]).repeat(BATCH, 1)
    batch["search_wdl"] = torch.tensor([[0.7, 0.2, 0.1]]).repeat(BATCH, 1)
    kwargs = {
        "sf_wdl_frac": 0.50, "search_wdl_frac": 0.20,
        "sf_search_dampen_sf_low": 1.0,
    }
    target = _captured_wdl_target(monkeypatch, batch, outputs, **kwargs)
    assert _decompose(target, batch)["outcome"] == pytest.approx(0.80, abs=1e-5)

    readout = _readout_from_compute_loss(batch, outputs, **kwargs)
    assert readout.sf_effective_rows == pytest.approx(0.0), (
        "every row is has_sf_wdl=1, so a LABEL count would read BATCH here and "
        "report a leak of 0.00 on a target that is 0.80 raw outcome"
    )
    assert readout.leaked_to_outcome == pytest.approx(0.50)
    with pytest.raises(ValueBlendMisconfigured):
        assert_no_silent_outcome_fallback(readout)


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
        sf_wdl_frac=0.50, search_wdl_frac=0.20,
        sf_effective_rows=0.0, search_effective_rows=512.0, batch_rows=512.0,
    )
    assert readout.leaked_to_outcome == pytest.approx(0.50)
    assert readout.outcome_borne_frac == pytest.approx(0.80)
    with pytest.raises(ValueBlendMisconfigured, match=re.escape("0.5000 of the WDL")):
        assert_no_silent_outcome_fallback(readout)


def test_guard_passes_on_the_override() -> None:
    """The CURRENT override: the whole live SF share handed to lc0's search."""
    readout = value_blend_readout(
        sf_wdl_frac=0.0, search_wdl_frac=1.0,
        sf_effective_rows=0.0, search_effective_rows=512.0, batch_rows=512.0,
    )
    assert readout.leaked_to_outcome == 0.0
    assert readout.outcome_borne_frac == pytest.approx(0.0)
    assert_no_silent_outcome_fallback(readout)


def test_the_superseded_0_70_override_is_now_refused() -> None:
    """⚑⚑ THE BAR MOVED 0.30 -> 0.00 AND THIS IS WHAT THAT MEANS.

    `sf 0.0 / search 0.70` was the lc0 control's shipped override until
    2026-08-16, justified as "keeps `game_frac` at production's 0.30". That
    0.30 came from `main`'s committed yaml; the LIVE blend is `0.69 + 0.31 =
    1.00`, so production's `game_frac` is ZERO and the old override was putting
    0.30 of the value target on the raw game outcome that production puts none
    on. Nothing leaked — `leaked_to_outcome` is exactly 0.0 here — which is
    precisely why `outcome_borne_frac` is the number that had to be gated.
    """
    readout = value_blend_readout(
        sf_wdl_frac=0.0, search_wdl_frac=0.70,
        sf_effective_rows=0.0, search_effective_rows=512.0, batch_rows=512.0,
    )
    assert readout.leaked_to_outcome == 0.0
    assert readout.outcome_borne_frac == pytest.approx(0.30)
    with pytest.raises(ValueBlendMisconfigured, match=re.escape("0.3000 of its mass")):
        assert_no_silent_outcome_fallback(readout)


def test_guard_passes_when_every_row_carries_a_label() -> None:
    """LIVE production's own shape must not trip the guard."""
    readout = value_blend_readout(
        sf_wdl_frac=0.69, search_wdl_frac=0.31,
        sf_effective_rows=512.0, search_effective_rows=512.0, batch_rows=512.0,
    )
    assert readout.leaked_to_outcome == 0.0
    assert readout.game_frac == pytest.approx(0.0)
    assert_no_silent_outcome_fallback(readout)


def test_the_bar_is_the_live_production_game_frac_not_a_literal() -> None:
    """⚑ `PRODUCTION_GAME_FRAC` is a hand-written constant in a module that
    cannot import the pin without a cycle, so the constant is pinned to the pin
    HERE instead. Without this it is exactly the kind of copied number that sat
    at `main`'s 0.30 for a day after production moved off it.
    """
    live = LIVE_TRAINER_PIN["kwargs"]
    expected = 1.0 - float(live["sf_wdl_frac"]) - float(live["search_wdl_frac"])
    assert pytest.approx(expected) == PRODUCTION_GAME_FRAC, (
        f"PRODUCTION_GAME_FRAC is {PRODUCTION_GAME_FRAC} but the LIVE blend "
        f"({live['sf_wdl_frac']} + {live['search_wdl_frac']}) leaves "
        f"{expected}. Regenerate LIVE_TRAINER_PIN and move the constant with "
        "it — the two are one measurement recorded twice."
    )


def test_guard_catches_a_partial_leak() -> None:
    """Half the rows labelled leaks half the SF share, and is not tolerated."""
    readout = value_blend_readout(
        sf_wdl_frac=0.50, search_wdl_frac=0.20,
        sf_effective_rows=256.0, search_effective_rows=512.0, batch_rows=512.0,
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


# ── the CATEGORICAL head: the same fallback, a different door ────────────────
#
# ⚑ The first two tests here do NOT test the guard. They test the CLAIM, through
# the production target builder, exactly as this file's first test does for the
# WDL head.

LIVE_CATEGORICAL = CategoricalTargetParams(
    blend_frac=0.69, search_blend_frac=0.31, num_bins=32, sigma=0.04,
)


def _categorical_arrays(*, with_sf_wdl: bool, with_target: bool) -> dict[str, Any]:
    """Sampled arrays shaped like the converter's output, `wdl_target == 0` (win).

    ⚑ `with_target=False` is the real converted lc0 row set:
    `scripts/lc0_data_to_rows.py` writes no `categorical_target` column, which
    is measured — a batch sampled from
    `data/lc0_rows/training-run2-test91-*` carries 19 keys and that is not one.
    """
    import numpy as np

    arrs: dict[str, Any] = {
        "wdl_target": np.zeros((4,), dtype=np.int8),
        "search_wdl": np.tile(np.array([0.5, 0.3, 0.2], dtype=np.float32), (4, 1)),
        "has_search_wdl": np.ones((4,), dtype=np.float32),
    }
    if with_target:
        arrs["categorical_target"] = np.zeros((4, 32), dtype=np.float32)
        arrs["categorical_target"][:, 31] = 1.0
    if with_sf_wdl:
        arrs["sf_wdl"] = np.tile(np.array([0.6, 0.3, 0.1], dtype=np.float32), (4, 1))
        arrs["has_sf_wdl"] = np.ones((4,), dtype=np.float32)
    return arrs


def test_the_rebuild_is_a_no_op_without_a_categorical_target_column() -> None:
    """⚑⚑ THIS IS WHY THE ARM MAY CARRY LIVE'S `rebuild_categorical_target`.

    Production runs the rebuild with `blend_frac 0.69`, and the lc0 control
    copies it verbatim because matching production is the arm's premise. It is
    safe only because `rebuild_categorical_target_in_arrays` returns the batch
    UNTOUCHED when there is no `categorical_target`, and converted lc0 rows
    have none — a property of the CONVERTER, not of the config, which is why
    the driver measures it per batch rather than citing this test.
    """
    arrs = _categorical_arrays(with_sf_wdl=False, with_target=False)
    out = rebuild_categorical_target_in_arrays(dict(arrs), params=LIVE_CATEGORICAL)
    assert "categorical_target" not in out
    assert set(out) == set(arrs)


def test_the_rebuild_moves_0_69_onto_the_outcome_when_sf_is_absent() -> None:
    """⚑⚑ THE ARMED MECHANISM, MEASURED — the WDL leak one head over.

    With no `sf_wdl`, `normalize_categorical_blend_fracs` DROPS `blend_frac`
    and does NOT redistribute it, so `game_frac` becomes 0.69 and the target
    the head is trained toward moves by `0.69 * (outcome - E[sf])`. Asserted as
    a NUMBER, not as a share: the HL-Gauss target is a distribution over bins
    and a share decomposition of it would not be linear.
    """
    import numpy as np

    search = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    sf = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    labelled = categorical_target_value(
        1.0, sf, blend_frac=0.69, search_wdl=search, search_blend_frac=0.31,
    )
    unlabelled = categorical_target_value(
        1.0, None, blend_frac=0.69, search_wdl=search, search_blend_frac=0.31,
    )
  # E[sf] = 0.6 - 0.1 = 0.5; E[search] = 0.5 - 0.2 = 0.3; outcome = +1.0.
    assert labelled == pytest.approx(0.69 * 0.5 + 0.31 * 0.3)
    assert unlabelled == pytest.approx(0.69 * 1.0 + 0.31 * 0.3)
    assert unlabelled - labelled == pytest.approx(0.69 * (1.0 - 0.5))
  # ...and it really does rewrite the stored target when the column exists.
    arrs = _categorical_arrays(with_sf_wdl=False, with_target=True)
    before = arrs["categorical_target"].copy()
    out = rebuild_categorical_target_in_arrays(dict(arrs), params=LIVE_CATEGORICAL)
    assert not np.array_equal(out["categorical_target"], before)


@pytest.mark.parametrize(
    ("sf_available", "search_available", "expected"),
    [
        (True, True, (0.69, 0.31, 0.0)),
        (False, True, (0.0, 0.31, 0.69)),
        (True, False, (0.69, 0.0, 0.31)),
        (False, False, (0.0, 0.0, 1.0)),
    ],
)
def test_normalize_categorical_blend_fracs_drops_without_redistributing(
    sf_available: bool, search_available: bool, expected: tuple[float, float, float],
) -> None:
    assert normalize_categorical_blend_fracs(
        0.69, 0.31, sf_available=sf_available, search_available=search_available,
    ) == pytest.approx(expected)


def test_the_extracted_normalisation_is_the_one_the_target_builder_uses() -> None:
    """A guard must share the criterion's instrument.

    Rather than assert the two call the same function, drive the
    OVER-SUBSCRIBED branch — where a hand-derived `1 - blend - search_blend`
    would be wrong — through `categorical_target_value` and check the value
    agrees with the fracs the helper reports.
    """
    import numpy as np

    search = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    sf = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    sf_frac, search_frac, game_frac = normalize_categorical_blend_fracs(
        0.8, 0.4, sf_available=True, search_available=True,
    )
    assert game_frac == 0.0
    assert categorical_target_value(
        1.0, sf, blend_frac=0.8, search_wdl=search, search_blend_frac=0.4,
    ) == pytest.approx(sf_frac * 0.5 + search_frac * 0.3)


def _categorical_readout(**overrides: Any) -> CategoricalRebuildReadout:
    fields: dict[str, Any] = {
        "rebuild_enabled": True, "blend_frac": 0.69, "search_blend_frac": 0.31,
        "target_present": True, "sf_labelled_frac": 1.0,
        "search_labelled_frac": 1.0, "batch_rows": 512.0,
    }
    fields.update(overrides)
    return CategoricalRebuildReadout(**fields)


def test_the_categorical_guard_passes_on_the_lc0_corpus() -> None:
    """No `categorical_target` column ⇒ the rebuild never runs ⇒ nothing moves."""
    readout = _categorical_readout(target_present=False, sf_labelled_frac=0.0)
    assert readout.applies is False
    assert readout.outcome_borne_frac == pytest.approx(0.0)
    assert_categorical_rebuild_is_inert(readout)


def test_the_categorical_guard_passes_on_production() -> None:
    """LIVE production's own shape — every label present — must not trip it."""
    readout = _categorical_readout()
    assert readout.applies is True
    assert readout.outcome_borne_frac == pytest.approx(0.0)
    assert_categorical_rebuild_is_inert(readout)


def test_the_categorical_guard_fires_if_the_corpus_gains_a_target() -> None:
    """⚑ The gate must be able to FAIL, and this is the state it exists for."""
    readout = _categorical_readout(sf_labelled_frac=0.0)
    assert readout.outcome_borne_frac == pytest.approx(0.69)
    with pytest.raises(ValueBlendMisconfigured, match=re.escape("0.6900 of its mass")):
        assert_categorical_rebuild_is_inert(readout)


def test_the_categorical_guard_is_linear_in_partial_label_coverage() -> None:
    """Half the rows labelled moves half the share — not all of it, not none."""
    readout = _categorical_readout(sf_labelled_frac=0.5)
    assert readout.outcome_borne_frac == pytest.approx(0.345)
    with pytest.raises(ValueBlendMisconfigured):
        assert_categorical_rebuild_is_inert(readout)


def test_the_categorical_guard_reports_zero_when_the_rebuild_is_off() -> None:
    """`rebuild_categorical_target: false` leaves the STORED target alone, so
    the configured fracs describe nothing and must not read as a failure."""
    readout = _categorical_readout(rebuild_enabled=False, sf_labelled_frac=0.0)
    assert readout.applies is False
    assert readout.outcome_borne_frac == pytest.approx(0.0)
    assert_categorical_rebuild_is_inert(readout)
