"""``policy_target_temp``: a temperature on the MAIN policy target.

Three properties are load-bearing, and each test below is written so that
DELETING the line it covers turns it red — an earlier revision of this file
asserted only that "the numbers changed", and six semantic mutants survived it.

1. **Identity at the default.** It ships default-off into a live training loop,
   so anything other than bit-identity at ``temp == 1.0`` is a silent
   production change.
2. **Zeros stay zero.** The moves this target zeroes were measured to lose a
   median 538cp; flattening must move mass WITHIN the support, never onto them.
3. **The eval ruler does not move with the arm.** CE's floor is the target's
   entropy, so retempering raises ``policy_ce`` for an unchanged model. If the
   holdout eval used the reshaped target, every arm-vs-baseline comparison
   would be two different rulers. ``Trainer._eval_loss_kwargs`` pins it off.
"""
from __future__ import annotations

import math

import pytest
import torch

from chess_anti_engine.train.losses import compute_loss, retemper_main_policy_target

WIDTH = 8


def _target(rows: list[list[float]]) -> torch.Tensor:
    t = torch.tensor(rows, dtype=torch.float32)
    return t / t.sum(dim=-1, keepdim=True)


def _entropy(p: torch.Tensor) -> float:
    return float(-(p * torch.log(p.clamp_min(1e-30))).sum())


def test_the_default_is_bit_identical_and_does_not_even_allocate() -> None:
    """THE CONTROL: shipping default-off must not perturb a single bit."""
    pol = _target([[5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert retemper_main_policy_target(pol, temp=1.0) is pol


def test_the_output_is_exactly_the_renormalised_power() -> None:
    """Pins the ARITHMETIC, not just its direction.

    A mutant that used ``temp`` instead of ``1/temp``, or that skipped the
    renormalise, or that overwrote the row with a constant, all still produce
    "a flatter-looking vector"; only an exact expectation rejects them.
    """
    pol = _target([[8.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = retemper_main_policy_target(pol, temp=1.30)
    expected = pol[0] ** (1.0 / 1.30)
    expected = expected / expected.sum()
    assert torch.allclose(out[0], expected, atol=1e-7)
    assert out[0].sum().item() == pytest.approx(1.0, abs=1e-6)


def test_zeroed_moves_stay_exactly_zero() -> None:
    """The support is not widened. Zero ** anything positive is zero, and the
    moves outside the support lose a median 538cp -- importing them is the
    opposite of the intent."""
    pol = _target([[8.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    for temp in (1.30, 2.0, 5.0, 0.70):
        out = retemper_main_policy_target(pol, temp=temp)
        assert (out[0, 3:] == 0.0).all(), f"temp {temp} lifted a zeroed move off zero"


def test_ordering_is_preserved_at_every_temperature() -> None:
    """A power with a positive exponent is monotone, so RANKING is invariant --
    this knob changes confidence only. A sign error in the exponent would
    invert the ranking while still summing to 1.0, so nothing downstream would
    complain."""
    pol = _target([[8.0, 5.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    order = torch.argsort(pol[0], descending=True)
    for temp in (0.5, 0.99, 1.01, 1.30, 3.0):
        out = retemper_main_policy_target(pol, temp=temp)
        assert torch.equal(torch.argsort(out[0], descending=True), order)


def test_direction_is_a_temperature_not_a_magnitude() -> None:
    """Negative control on direction: >1 flattens, <1 sharpens, monotonically."""
    pol = _target([[8.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    ents = [_entropy(retemper_main_policy_target(pol, temp=t)[0])
            for t in (0.70, 1.0, 1.30, 2.0)]
    assert ents == sorted(ents), f"entropy must rise monotonically with temp: {ents}"
    assert ents[1] == pytest.approx(_entropy(pol[0]), abs=1e-6)


def test_an_all_zero_row_stays_all_zero_instead_of_going_nan() -> None:
    """Padding / masked-out rows carry an all-zero target. Without the
    ``clamp_min`` on the denominator this is 0/0 -> NaN, and a NaN target
    NaNs the whole batch's loss."""
    pol = torch.zeros(1, WIDTH)
    out = retemper_main_policy_target(pol, temp=1.30)
    assert torch.isfinite(out).all()
    assert (out == 0.0).all()


def test_a_negative_input_cannot_produce_a_nan_target() -> None:
    """``clamp_min(0.0)`` is load-bearing: a fractional power of a negative
    number is NaN in torch, and a stored target is not guaranteed non-negative
    by any check upstream of here."""
    pol = torch.tensor([[0.5, -0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = retemper_main_policy_target(pol, temp=1.30)
    assert torch.isfinite(out).all(), "a negative entry reached a fractional power"
    assert (out >= 0.0).all()


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.5, float("nan"), float("inf")])
def test_a_hostile_temperature_is_refused_at_the_boundary(bad: float) -> None:
    """None of these fails loudly on its own: 0.0 divides by zero INSIDE the
    training step, a negative value inverts the ordering while still summing
    to 1.0, nan/inf propagate silently. The key is live-reloadable-adjacent, so
    a yaml typo must not reach the loss."""
    pol = _target([[4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="policy_target_temp"):
        retemper_main_policy_target(pol, temp=bad)


def test_a_batch_is_retempered_per_row_not_jointly() -> None:
    """The renormalise must be per-row. A missing ``dim=-1`` would divide every
    row by the batch total, so each row would sum to something other than 1 and
    the effective per-row loss weight would silently depend on batch content."""
    pol = _target([[8.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    out = retemper_main_policy_target(pol, temp=1.30)
    assert torch.allclose(out.sum(-1), torch.ones(2), atol=1e-6)
    solo = retemper_main_policy_target(pol[:1], temp=1.30)
    assert torch.allclose(out[0], solo[0], atol=1e-7), "row 0 depends on row 1"


def test_a_uniform_target_is_a_fixed_point() -> None:
    """Sanity anchor: there is no confidence to remove from a uniform target,
    so the knob must be a no-op there at every temperature."""
    pol = _target([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    for temp in (0.5, 1.30, 4.0):
        assert torch.allclose(retemper_main_policy_target(pol, temp=temp), pol, atol=1e-6)


class _Head(torch.nn.Module):
    """Emits FIXED logits equal to log(target), i.e. a model that already
    predicts the stored target exactly. Its CE is then exactly the target's
    entropy, which makes the ruler-shift below arithmetic rather than a trend."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = int(x.shape[0])
        return {
            "policy_own": self.logits.expand(n, -1).clone(),
            "wdl": torch.zeros(n, 3),
        }


def test_retempering_moves_the_ce_floor_which_is_why_eval_pins_it_off() -> None:
    """THE RULER TEST. Same model, same rows, two temperatures -- ``policy_ce``
    moves by ~0.6 nats. That is the target's entropy changing, not the model.

    This is the property that makes sharing ``_loss_kwargs`` between the
    training step and the holdout eval a defect: an arm trained at temp 1.3
    would report a worse CE than its control while being an identical model,
    and the offline rig prints exactly that number.
    """
    pol = _target([[0.55, 0.25, 0.12, 0.08, 0.0, 0.0, 0.0, 0.0]])
    model = _Head(torch.log(pol[0].clamp_min(1e-30)))
    batch = {
        "x": torch.zeros(1, 4),
        "policy_t": pol,
        "wdl_t": torch.zeros(1, dtype=torch.long),
        "legal_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32),
        "has_legal_mask": torch.ones(1),
    }
    out = model(batch["x"])

    pure = float(compute_loss(out, batch, policy_target_temp=1.0)["policy_ce"].mean())
    warm = float(compute_loss(out, batch, policy_target_temp=1.30)["policy_ce"].mean())

    assert pure == pytest.approx(_entropy(pol[0]), abs=1e-4), (
        "a model emitting log(target) should read CE == the target's entropy"
    )
    assert warm > pure + 0.05, (
        f"retempering must move the CE floor for it to be a ruler hazard: "
        f"{pure:.4f} -> {warm:.4f}"
    )
    flat = retemper_main_policy_target(pol, temp=1.30)
    assert warm == pytest.approx(
        -float((flat[0] * torch.log_softmax(out["policy_own"][0], dim=-1)).sum()), abs=1e-4,
    )
    assert not math.isclose(warm, pure), "the arms would be indistinguishable"


def test_the_eval_path_is_wired_to_the_pinned_kwargs() -> None:
    """The test above proves the HAZARD; this one proves the fix is on the path.

    Read structurally rather than by calling ``_compute_metrics``, which needs a
    constructed Trainer, a model and a replay buffer. Both ``compute_loss`` call
    sites are named, so a future edit that points the eval at ``_loss_kwargs``
    (or adds a third call site with neither) fails here.
    """
    import ast
    import inspect

    from chess_anti_engine.train import trainer as trainer_mod

    tree = ast.parse(inspect.getsource(trainer_mod))
    sites = {
        fn.name: [ast.unparse(k.value) for k in call.keywords if k.arg is None]
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for call in ast.walk(fn)
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "compute_loss"
    }
    assert sites == {
        "_compute_metrics": ["self._eval_loss_kwargs"],
        "_run_optimizer_step": ["self._loss_kwargs"],
    }, f"compute_loss call sites moved: {sites}"


def test_the_pinned_kwargs_override_only_the_target_shape() -> None:
    """``_eval_loss_kwargs`` must pin the reshape and change NOTHING else --
    pinning a loss weight would make eval's ``total`` stop matching the trained
    objective, which is the same ruler-drift defect in the other direction."""
    import inspect

    from chess_anti_engine.train.trainer import Trainer

    stub = type("S", (), {
        "_loss_kwargs": {"policy_target_temp": 1.30, "w_policy": 1.0, "w_sf_own": 0.1},
    })()
    prop = inspect.getattr_static(Trainer, "_eval_loss_kwargs")
    assert isinstance(prop, property)
    assert prop.fget is not None
    pinned = prop.fget(stub)
    assert pinned["policy_target_temp"] == 1.0
    assert {k: v for k, v in pinned.items() if k != "policy_target_temp"} == {
        "w_policy": 1.0, "w_sf_own": 0.1,
    }
