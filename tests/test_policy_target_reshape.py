"""``policy_target_temp``: a temperature on the MAIN policy target.

Three properties are load-bearing, and each test below is written so that
DELETING the line it covers turns it red — an earlier revision of this file
asserted only that "the numbers changed", and six semantic mutants survived it.

1. **Identity at the default.** It ships default-off into a live training loop,
   so anything other than bit-identity at ``temp == 1.0`` is a silent
   production change.
2. **Zeros stay zero.** The moves this target zeroes were measured to lose a
   median 538cp; flattening must move mass WITHIN the support, never onto them.
3. **The eval ruler does not move with the arm.** ``CE = H(target) +
   KL(target||model)``, and retempering changes both, so an UNCHANGED model
   reads a different ``policy_ce``. The direction is model-dependent -- it has
   been measured falling on a random-logit fixture -- so the tests below assert
   that the ruler MOVES, and quote a magnitude only for the one fixture where
   the sign is forced (a perfectly-fit model, where KL starts at 0 and can only
   rise). If the holdout eval used the reshaped target, every arm-vs-baseline
   comparison would be two different rulers. ``Trainer._eval_loss_kwargs``
   pins it off.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from chess_anti_engine.train.losses import (
    _POLICY_TARGET_TEMP_MAX,
    compute_loss,
    policy_target_temp_active,
    retemper_main_policy_target,
)
from chess_anti_engine.train.trainer import Trainer

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
    for temp in (1.30, 2.0, 4.0, 0.70):
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
    moves. That is the target changing, not the model.

    ⚑ REVIEW NOTE 4: the DIRECTION is not a property of the transform.
    ``CE = H(target) + KL(target||model)``; flattening raises ``H`` but can
    lower ``KL`` by more, and on a random-logit fixture `policy_ce` was measured
    to FALL. The sign is forced HERE, and only here, because the model is
    constructed to emit ``log(target)`` exactly: ``KL`` starts at 0, so it can
    only rise, and ``H`` rises too. Any magnitude quoted for this knob has to
    name the model and fixture it came from -- an earlier revision of the
    docstrings stated "~+0.62 nats at temp 1.3" unconditionally, which is a
    property of a well-fit model on a sharp target rather than of the code.

    The eval pin needs only the weaker claim, which is what makes sharing
    ``_loss_kwargs`` between the training step and the holdout eval a defect:
    an arm trained at temp 1.3 reports a DIFFERENT CE from its control while
    being an identical model, and the offline rig prints exactly that number.
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
  # `>` and not `!=`: on THIS fixture the model is exactly log(target), so KL
  # is 0 at temp 1.0 and strictly positive at 1.3 while H also rises. The
  # inequality is therefore a real prediction here -- it is not a claim that
  # retempering raises CE in general.
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

    ⚑ The eval site now stars a LOCAL (``eval_loss_kwargs``), because the async
    path scores a snapshot under the objective captured at ``start()``. Pinning
    the local's NAME alone would make this test vacuous for its own regression --
    a local can be bound to anything, ``self._loss_kwargs`` included. So the
    binding SET is pinned too, and both branches must resolve to a pinned object.
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
        "_compute_metrics": ["eval_loss_kwargs"],
        "_run_optimizer_step": ["self._loss_kwargs"],
    }, f"compute_loss call sites moved: {sites}"

    metrics_fn = next(
        fn for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and fn.name == "_compute_metrics"
    )

    def _bindings(name: str) -> set[str]:
        return {
            ast.unparse(node.value)
            for node in ast.walk(metrics_fn)
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == name for t in node.targets)
        }

  # ⚑ ALL THREE LOCALS, not just the kwargs. `ObjectiveSnapshot` carries the
  # loss AND the two ruler inputs precisely so they cannot come from different
  # logical times; pinning only the kwargs here would let a ruler leg drift back
  # to a live read and still pass, and the result would be a `test_loss`
  # measured under objective A stamped with objective B's identity -- internally
  # consistent-looking and wrong. `tests/test_holdout_ruler_identity.py::
  # test_a_pinned_objective_survives_a_live_flip_in_BOTH_legs` executes the
  # same three legs; this one keeps the CALL SITES honest.
    for local, pinned, live in (
        ("eval_loss_kwargs", "objective.loss_kwargs", "self._eval_loss_kwargs"),
        ("ruler_weights", "objective.loss_weights", "self._ruler_loss_weights()"),
        ("ruler_shape", "objective.loss_shape", "self._ruler_loss_shape()"),
    ):
        assert _bindings(local) == {live, pinned}, (
            f"`{local}` must resolve to a PINNED object on the objective branch "
            f"and to live state only when `objective is None`: {_bindings(local)}"
        )


def test_the_pinned_kwargs_override_only_the_target_shape_and_the_row_selection() -> None:
    """``_eval_loss_kwargs`` pins what CHANGES A COLUMN'S DEFINITION, and nothing else.

    ⚑⚑ THIS TEST'S RULE WAS DELIBERATELY REVISED, AND THE TENSION IS REAL -- read this
    before "fixing" it in either direction. The original rule was "pin the reshape and
    change NOTHING else", justified as: *pinning a loss weight would make eval's
    ``total`` stop matching the trained objective, which is the same ruler-drift defect
    in the other direction.* That concern is correct and still applies to WEIGHTS. A
    weight scales a term over the rows already in the window, so eval measuring the
    unweighted term is a straightforward mismatch with training.

    The two ``sf_own_regret`` gate keys are NOT weights, and that is why they are pinned:
    the gate applies **per row, on a data-dependent predicate**, so it changes WHICH
    rows the ``sf_own_regret`` column is computed over. It redefines the column rather
    than scaling it. Unpinned, the measured cost is not subtle -- an **unchanged model**
    reads ``sf_own_regret`` 0.4174 -> 0.2112, a 2x move produced entirely by a training
    knob, so arming the arm would look like the eval loss improving with zero learning.

    ⚑ THE PRICE, stated rather than hidden: while the arm is armed, eval's ``total``
    genuinely is NOT the trained objective for this one term -- exactly the defect the
    original docstring named. We take that trade because eval's job here is to be a
    STABLE RULER across arm and baseline, and a ruler that moves with the intervention
    cannot compare them. ``test_loss``, the best-model handover and the promotion
    comparison all read this number.
    ⚑ The divergence is EXACTLY ZERO today: ``w_sf_own_regret: 0.0`` on the live yaml,
    so the term contributes nothing to ``total`` either way. This is a
    change-before-ARMING, not a change-before-merging.

    ⚑ It also closes a blind spot at the root instead of by widening a digest:
    ``eval_ruler_id`` hashes the SOURCE of the covered frames, so it moves when
    ``compute_loss`` is edited but is BLIND to ``sf_regret_gate_scale``, the helper that
    decides the number. Pinned off, that helper cannot reach the eval measurement.

    So the invariant is not "only the target shape" but: **every pinned key must
    redefine a column, and no pinned key may be a plain loss weight.** A future key that
    merely scales a term belongs in ``_loss_kwargs`` only.
    """
    import inspect

    from chess_anti_engine.train.trainer import Trainer

    stub = type("S", (), {
        "_loss_kwargs": {
            "policy_target_temp": 1.30,
            "w_policy": 1.0,
            "w_sf_own": 0.1,
            "sf_own_regret_listed_mass_min": 0.8,
            "sf_own_regret_unlisted_scale": 0.25,
        },
    })()
    prop = inspect.getattr_static(Trainer, "_eval_loss_kwargs")
    assert isinstance(prop, property)
    assert prop.fget is not None

    # ⚑⚑ NON-DEGENERACY FIRST, and it is not ceremony: a mutant that deleted the two
    # gate keys from the stub above left this whole file GREEN. `_eval_loss_kwargs` is
    # `{**self._loss_kwargs, <pins>}`, so it ADDS the keys whether or not the trainer
    # had them -- and then asserting `pinned[key] == <identity>` passes without ever
    # proving the pin OVERRODE anything. The stub's values must be non-identity, and
    # that must be asserted here, or the override is untested.
    train_side = stub._loss_kwargs  # pyright: ignore[reportAttributeAccessIssue]
    for key, identity in (
        ("policy_target_temp", 1.0),
        ("sf_own_regret_listed_mass_min", 0.0),
        ("sf_own_regret_unlisted_scale", 1.0),
    ):
        assert key in train_side, f"stub lost {key}; the override would be untested"
        assert train_side[key] != identity, (
            f"stub's {key} is already the identity, so pinning it proves nothing"
        )

    pinned = prop.fget(stub)
    # Target SHAPE pinned to the identity.
    assert pinned["policy_target_temp"] == 1.0
    # Row SELECTION pinned to the identity: mass_min 0.0 can match no row (policy mass
    # is non-negative) and scale 1.0 downweights nothing even if one did.
    assert pinned["sf_own_regret_listed_mass_min"] == 0.0
    assert pinned["sf_own_regret_unlisted_scale"] == 1.0
    # ⚑ Everything else passes through UNTOUCHED -- in particular `w_sf_own_regret`
    # itself, which is a weight. If a later edit starts pinning weights too, this fails.
    assert {
        k: v for k, v in pinned.items()
        if k not in {
            "policy_target_temp",
            "sf_own_regret_listed_mass_min",
            "sf_own_regret_unlisted_scale",
        }
    } == {"w_policy": 1.0, "w_sf_own": 0.1}


# ── The underflow the max-scaling makes impossible ──────────────────────────


@pytest.mark.parametrize("temp", [0.5, 0.9, 1.3, 2.2, 4.0])
def test_no_temperature_can_empty_the_target(temp: float) -> None:
    """⚑ REVIEW FINDING B1, and the reason for the max-scaling.

    `policy_target_temp: 0.001` -- a plausible typo -- used to drive every entry
    of a broad target to 0 in fp32 via `p ** 1000`. The renormalise divided by a
    clamped denominator instead of raising, so the target was ALL-ZERO,
    `policy_ce` was exactly 0.0, and the policy head trained on nothing while
    the loss read as PERFECT. The holdout could not see it either, because eval
    is pinned to temp 1.0.

    The naive form fails here by construction: the broadest realistic target is
    ~uniform over the 1858 compact moves, and (1/t)*log10(1858) exceeds fp32's
    38-decade normal range for any t below ~0.086.
    """
    broad = torch.full((1, 1858), 1.0 / 1858)
    naive = broad ** (1.0 / temp)

    out = retemper_main_policy_target(broad, temp=temp)
    assert torch.isfinite(out).all()
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-5), (
        f"temp={temp} produced a target summing to {float(out.sum())!r}; the "
        f"naive p**(1/t) summed to {float(naive.sum()):.3e}"
    )
    assert float(out.max()) > 0.0, "the target has no mass anywhere"


@pytest.mark.parametrize("bad", [0.001, 0.05, 0.49])
def test_a_temperature_below_the_floor_is_refused(bad: float) -> None:
    """The floor is a typo catcher, not a numerical limit -- the max-scaling
    already makes these safe. It exists because over-SHARPENING fails quietly:
    a one-hot target has entropy 0, so `policy_ce` falls, and the mistake reads
    as the loss improving."""
    with pytest.raises(ValueError, match="policy_target_temp must be finite"):
        retemper_main_policy_target(torch.ones(1, 4) / 4, temp=bad)


@pytest.mark.parametrize("bad", [4.01, 10.0, 15.0, 150.0, 1000.0, 1e30])
def test_a_temperature_above_the_ceiling_is_refused(bad: float) -> None:
    """⚑ REVIEW FINDING N3, and the reason the ceiling exists at all.

    There was no upper bound, so `policy_target_temp: 15` -- a dropped decimal
    point on the screen's own arm value of 1.5 -- was ACCEPTED, and it drives
    the target to uniform over its support (`p ** (1/15)`).

    The docstring's mitigation for that was "over-flattening raises `policy_ce`
    loudly and gets noticed". It does not, and the reason is this PR's own
    design: `Trainer._eval_loss_kwargs` PINS eval to 1.0, so the holdout
    `policy_ce` an operator watches is invariant to the arm's temperature by
    construction. Only the train-side `policy_loss` moves -- on an arm launched
    precisely because it was expected to move. So the typo is silently wrong,
    and a range check is the only thing between it and a 21-hour sweep.

    `1e30` also pins that the range check subsumes the old explicit `+inf`
    test rather than merely sitting beside it.
    """
    with pytest.raises(ValueError, match="policy_target_temp must be finite"):
        retemper_main_policy_target(torch.ones(1, 4) / 4, temp=bad)


def test_the_ceiling_clears_every_value_anyone_would_deliberately_set() -> None:
    """The other half of the mutation: a ceiling of 1.0 would also pass the test
    above and would refuse the knob this PR exists for. Pin the values that are
    load-bearing SOMEWHERE ELSE in the tree, so lowering the bound breaks here
    instead of at launch.

    * 1.5   -- the offline screen's arm in `docs/experiment_ledger.md`
    * 2.2   -- the top of the 1.36-2.20 band lc0's `--policy-softmax-temp` has
      run at. ⚑ That is a SEARCH-TIME PRIOR temperature, the analogue of our
      `gumbel_policy_temp` (production 1.5), NOT a training-target temperature
      like the knob under test. It is pinned here only as a magnitude a person
      deliberately sets, so a lowered ceiling breaks visibly.
    * 0.5 / 2.0 -- `scripts/retarget_retrain.py`'s reachability probe values.
    """
    for good in (0.5, 1.36, 1.5, 2.0, 2.2, 4.0):
        out = retemper_main_policy_target(torch.ones(1, 4) / 4, temp=good)
        assert torch.isfinite(out).all()
    assert _POLICY_TARGET_TEMP_MAX >= 2.2, (
        "the ceiling now sits inside the lc0 --policy-softmax-temp band the "
        "guard's comment clears, and below the 2.2 arm "
        "`test_a_yaml_temperature_REACHES_the_trainer...` launches"
    )


def test_the_max_scaling_does_not_change_the_result_where_it_was_already_safe() -> None:
    """The scaling must be a NUMERICS fix, not a behaviour change. A power
    transform is scale-free, so dividing by a positive constant first cannot
    move the output -- assert that rather than trusting the algebra."""
    peaked = torch.tensor([[0.7, 0.2, 0.07, 0.03]])
    for temp in (0.5, 1.3, 2.2):
        naive = peaked ** (1.0 / temp)
        naive = naive / naive.sum(dim=-1, keepdim=True)
        assert torch.allclose(
            retemper_main_policy_target(peaked, temp=temp), naive, atol=1e-6,
        ), f"temp={temp}: the max-scaling changed the result"


class _TinyPolicyModel(torch.nn.Module):
    """Smallest thing `Trainer.__init__` accepts -- the constructor only needs
    parameters to build optimizer groups from, and the validation under test
    runs before anything touches the model."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32, device=self.head.weight.device),
        }


def _trainer(tmp_path: Path, **kwargs: Any) -> Trainer:
    return Trainer(
        _TinyPolicyModel(), device="cpu", lr=1e-3, optimizer="adamw",
        use_amp=False, log_dir=tmp_path, tb_log_interval=1000,
        prefetch_batches=False, **kwargs,
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, 0.001, float("inf"), float("nan")])
def test_the_trainer_refuses_a_bad_temperature_at_CONSTRUCTION(
    tmp_path: Path, bad: float,
) -> None:
    """⚑ MUTATION SURVIVOR (review N3, and again at the #375 merge): the
    previous version of this test only `inspect.getsource`-grepped
    `Trainer.__init__` for the string ``retemper_main_policy_target(``.

    A grep for a call is not an observation of its EFFECT. Mutating the call's
    argument from ``temp=self.policy_target_temp`` to ``temp=1.0`` -- which
    validates a constant and therefore accepts every yaml value there is --
    left the whole file green (28 passed), because the string was still
    present and the separate helper-rejects-bad-values loop below still ran on
    a directly-supplied bad temp. The knob was accepted and silently ignored,
    which is this codebase's signature defect.

    So construct the real `Trainer` and require the bad value to be REFUSED
    THERE. Its comment promises the trial fails at STARTUP rather than inside
    the first training step, once the GPU is taken; that promise is now
    enforced by the only thing that can enforce it.
    """
    with pytest.raises(ValueError, match="policy_target_temp must be finite"):
        _trainer(tmp_path, policy_target_temp=bad)


def test_the_trainer_accepts_a_good_temperature_and_it_reaches_compute_loss(
    tmp_path: Path,
) -> None:
    """The other half of the mutation: refusing everything would also pass the
    test above. A good value must survive construction AND arrive at the
    training-step `compute_loss` call -- while the eval ruler stays pinned."""
    t = _trainer(tmp_path, policy_target_temp=1.30)

    assert t.policy_target_temp == pytest.approx(1.30)
    assert t._loss_kwargs["policy_target_temp"] == pytest.approx(1.30)
    assert t._eval_loss_kwargs["policy_target_temp"] == 1.0


def test_the_validation_helper_rejects_rather_than_being_an_inert_call_site(
) -> None:
    """The delegated check itself. 0.0 divides by zero, a negative exponent
    inverts the target's ordering while still summing to 1, and non-finite
    values propagate silently."""
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="policy_target_temp must be finite"):
            retemper_main_policy_target(torch.ones(1, 2) / 2, temp=bad)


# ── The yaml -> Trainer seam ────────────────────────────────────────────────
#
# ⚑ REVIEW FINDING 1 (independent review #2). Every test above hands the
# temperature straight to `Trainer(...)` as a kwarg, so the ONE seam that
# carries it in production -- the yaml -- had no coverage at all. The reviewer
# mutated `trainer_kwargs_from_config` to keep the literal `config.get` (so the
# startup-only derivation instrument still sees the read) while DISCARDING its
# value, and 180 tests passed. That instrument observes that a read EXISTS,
# never that its value is USED, which is a grep for a call standing in for an
# observation of its effect -- the same defect the constructor test above was
# rewritten to close, one layer further out.
#
# The failure it leaves open is the expensive one: `policy_target_temp: 1.5`
# validates, survives `_check_unknown`, reads back correctly from the trial's
# own config dict and from `scripts/audit_realized_config.py` (which reads the
# overlay, not the consumer) -- and the arm trains at 1.0. The null result gets
# attributed to the hypothesis instead of to the wiring.


def _flat_from_yaml_train_section(**train_overrides: object) -> dict[str, object]:
    """The production yaml, with `train:` overrides, through the real flattener."""
    import yaml as _yaml

    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    repo = Path(__file__).resolve().parent.parent
    cfg = _yaml.safe_load((repo / "configs" / "pbt2_small.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("train", {}).update(train_overrides)
    return flatten_run_config_defaults(cfg)


@pytest.mark.parametrize("temp", [1.30, 2.20])
def test_a_yaml_temperature_REACHES_the_trainer_and_the_training_step(
    tmp_path: Path, temp: float,
) -> None:
    """The whole chain, on the production config: yaml `train:` section ->
    `flatten_run_config_defaults` -> `trainer_kwargs_from_config` -> `Trainer`
    -> the `_loss_kwargs` that `_run_optimizer_step` splats into `compute_loss`.

    Each hop is asserted separately so a break says WHICH hop dropped it.

    ⚑ TWO VALUES, and 2.20 is not decoration (review #2, S1). Pinning a single
    value only proves the chain is identity AT that value: a silent
    `min(1.5, ...)` clamp inside `trainer_kwargs_from_config` passed a
    1.30-only test with 176 green, while truncating 2.20 -- which is inside the
    range the guard in `losses.py` is explicitly sized to accept (the 1.36-2.20
    lc0 `--policy-softmax-temp` band, a SEARCH-TIME PRIOR temperature cited
    there as a magnitude, not as a target temperature). A clamp is exactly the
    "accepted then quietly altered" shape, and one sample point cannot see it.

    ⚑ THE CTOR DICT IS SPLATTED WHOLE (review #2, S2). Production does
    `Trainer(model, model_config=..., **trainer_ctor)` at
    tune/trainable.py:759. An earlier version of this test re-extracted the one
    key it cared about and hand-built a Trainer from it, which left the SPLAT
    uncovered: `trainer_ctor.pop("policy_target_temp", None)` immediately
    before that line survived with 176 green. Only device/lr/amp and the
    prefetch thread are overridden here, for a CPU test process.
    """
    from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config

    flat = _flat_from_yaml_train_section(policy_target_temp=temp)
    assert flat["policy_target_temp"] == pytest.approx(temp), "the flattener dropped it"

    ctor = trainer_kwargs_from_config(
        {**flat, "device": "cpu", "no_amp": True, "lr": 1e-3}, log_dir=tmp_path,
    )
    assert ctor["policy_target_temp"] == pytest.approx(temp), (
        "trainer_kwargs_from_config read the key but did not carry its VALUE "
        f"unchanged at {temp} (got {ctor['policy_target_temp']!r}); the arm "
        "would train at some other temperature while its config reads correct"
    )

    ctor["prefetch_batches"] = False
    t = Trainer(_TinyPolicyModel(), **ctor)

    assert t.policy_target_temp == pytest.approx(temp)
    assert t._loss_kwargs["policy_target_temp"] == pytest.approx(temp)
  # ...and the ruler still does not move with the arm at the far end of the chain.
    assert t._eval_loss_kwargs["policy_target_temp"] == 1.0


def test_the_production_yaml_carries_no_temperature_today(tmp_path: Path) -> None:
    """The negative half: a value that arrives must have COME from the yaml.

    Asserting only the set cases would also pass if `trainer_kwargs_from_config`
    hardcoded one of them. The shipped config has no `policy_target_temp`, so
    the same chain must yield exactly the identity default -- which is also the
    claim that merging this PR cannot perturb the live run.
    """
    from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config

    import yaml as _yaml

    repo = Path(__file__).resolve().parent.parent
    raw = _yaml.safe_load((repo / "configs" / "pbt2_small.yaml").read_text(encoding="utf-8"))
    assert "policy_target_temp" not in (raw.get("train") or {}), (
        "the production config now sets policy_target_temp -- that is a live "
        "training-target change and needs a ledger entry, not a test update"
    )

    flat = _flat_from_yaml_train_section()
    assert "policy_target_temp" not in flat

    ctor = trainer_kwargs_from_config(
        {**flat, "device": "cpu", "no_amp": True, "lr": 1e-3}, log_dir=tmp_path,
    )
    assert ctor["policy_target_temp"] == 1.0
    ctor["prefetch_batches"] = False
    assert Trainer(_TinyPolicyModel(), **ctor)._loss_kwargs["policy_target_temp"] == 1.0


def test_a_bad_yaml_temperature_fails_at_STARTUP_through_the_real_chain(
    tmp_path: Path,
) -> None:
    """The constructor guard, reached the way an operator would reach it --
    through the same whole-dict splat production uses."""
    from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config

    ctor = trainer_kwargs_from_config(
        {**_flat_from_yaml_train_section(policy_target_temp=0.001),
         "device": "cpu", "no_amp": True, "lr": 1e-3},
        log_dir=tmp_path,
    )
    ctor["prefetch_batches"] = False
    with pytest.raises(ValueError, match="policy_target_temp must be finite"):
        Trainer(_TinyPolicyModel(), **ctor)


# ── Second effect: the temperature re-gates the policy_soft head ────────────
#
# ⚑ REVIEW NOTE 3 (independent review #2). `compute_loss` reassigns `pol_target`
# to the RETEMPERED target before the `soft_policy_min_tv` gate is computed from
# it, so the temperature also decides which rows the `policy_soft` head trains
# on. Latent in production (`soft_policy_min_tv: 0.0` in pbt2_small.yaml, and
# the knob appears only in configs/exp_soft_policy_divergent_only.yaml), but
# real the day the two are on together -- and asymmetric, because
# `_eval_loss_kwargs` pins the temperature and NOT `soft_policy_min_tv`, so
# training and eval would mask different row sets for that head.


def _soft_gate_batch(hard: torch.Tensor, soft: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "x": torch.zeros((1, 1, 1, 1)),
        "policy_t": hard,
        "policy_soft_t": soft,
        "has_policy": torch.ones((1,)),
        "has_policy_soft": torch.ones((1,)),
        "is_network_turn": torch.ones((1,)),
        "wdl_t": torch.zeros((1,), dtype=torch.long),
    }


@pytest.mark.parametrize(("temp", "expected_kept"), [(1.0, 1.0), (1.3, 0.0)])
def test_the_temperature_regates_the_soft_policy_head(
    temp: float, expected_kept: float,
) -> None:
    """The gate compares the SOFT target against the retempered HARD one, so
    moving the hard one moves the TV and flips the mask.

    The fixture makes the soft target a retempering of the hard target -- which
    is the case the gate exists for -- and puts the threshold between the two
    TVs, so nothing but `policy_target_temp` differs between the arms.
    """
    hard = _target([[0.55, 0.25, 0.12, 0.08]])
    soft = retemper_main_policy_target(hard, temp=1.25)

    tv_at = {
        t: float(0.5 * (retemper_main_policy_target(hard, temp=t) - soft).abs().sum())
        for t in (1.0, 1.3)
    }
    threshold = 0.5 * (tv_at[1.0] + tv_at[1.3])
    assert tv_at[1.3] < threshold < tv_at[1.0], (
        f"fixture no longer straddles the threshold: {tv_at}"
    )

    out = {
        "policy": torch.zeros((1, 4)),
        "policy_soft": torch.zeros((1, 4)),
        "wdl": torch.zeros((1, 3)),
    }
    losses = compute_loss(
        out, _soft_gate_batch(hard, soft),
        soft_policy_min_tv=threshold, policy_target_temp=temp,
        w_sf_move=0.0, w_sf_eval=0.0, w_categorical=0.0,
        w_volatility=0.0, w_moves_left=0.0,
    )
    assert float(losses["soft_mask_kept_frac"]) == pytest.approx(expected_kept), (
        "policy_target_temp changed which rows the policy_soft head trains on"
    )


def test_the_soft_gate_is_untouched_when_the_temperature_is_the_default() -> None:
    """THE CONTROL for the note above: default-off must not re-gate anything.

    Without this, the parametrised test would also pass if the gate were broken
    in some temperature-independent way."""
    hard = _target([[0.55, 0.25, 0.12, 0.08]])
    soft = retemper_main_policy_target(hard, temp=1.25)
    out = {
        "policy": torch.zeros((1, 4)),
        "policy_soft": torch.zeros((1, 4)),
        "wdl": torch.zeros((1, 3)),
    }
    batch = _soft_gate_batch(hard, soft)

  # Written out twice rather than splatted: not passing the key at all is
  # `main`'s behaviour (which has no such parameter), and passing it at its
  # default must be the same gate.
    absent = float(compute_loss(
        out, batch, soft_policy_min_tv=0.03, w_sf_move=0.0, w_sf_eval=0.0,
        w_categorical=0.0, w_volatility=0.0, w_moves_left=0.0,
    )["soft_mask_kept_frac"])
    explicit = float(compute_loss(
        out, batch, soft_policy_min_tv=0.03, w_sf_move=0.0, w_sf_eval=0.0,
        w_categorical=0.0, w_volatility=0.0, w_moves_left=0.0,
        policy_target_temp=1.0,
    )["soft_mask_kept_frac"])

    assert absent == pytest.approx(explicit)
    assert absent == pytest.approx(1.0), (
        "the fixture's TV is above the threshold at the default, so the "
        "default-off arm must keep the row"
    )


def test_the_production_ctor_dict_reaches_Trainer_UNMODIFIED() -> None:
    """⚑ REVIEW #2, S2. The test above splats a ctor dict, but it is a dict the
    TEST built. Production builds its own at tune/trainable.py and splats that
    one, and nothing observed the gap between the two: inserting
    ``trainer_ctor.pop("policy_target_temp", None)`` immediately before
    ``Trainer(model, model_config=model_cfg, **trainer_ctor)`` survived the
    whole battery, because no test executes that line.

    ``train_trial`` is a Ray entry point -- it wants a cluster, a GPU, a replay
    buffer and a checkpoint before it reaches this statement -- so there is no
    runtime observation available at unit-test cost. This asserts the invariant
    on the source instead, and deliberately asserts an ABSENCE OF TAMPERING
    rather than the presence of a call: the thing that went wrong last time was
    a grep for a call standing in for its effect, and a decorative `.pop` would
    satisfy a presence check while defeating the wiring.

    The invariant: the name bound by ``trainer_kwargs_from_config(...)`` is
    splatted into ``Trainer(...)`` with ``**``, and BETWEEN those two points
    nothing rebinds it, deletes from it, subscript-assigns into it, or calls a
    mutating method on it. Any of those is a knob silently dropped on the way
    to the consumer.
    """
    import ast
    import inspect

    from chess_anti_engine.tune import trainable as trainable_mod

    tree = ast.parse(inspect.getsource(trainable_mod))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "train_trial"
    )
    body = list(ast.walk(fn))

  # 1. Find the binding `<name> = trainer_kwargs_from_config(...)`.
    binds = [
        n for n in body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", None) == "trainer_kwargs_from_config"
    ]
    assert len(binds) == 1, f"expected exactly one ctor-dict binding, found {len(binds)}"
    assert isinstance(binds[0].targets[0], ast.Name)
    ctor_name = binds[0].targets[0].id
    bind_line = binds[0].lineno

  # 2. Find `Trainer(..., **<name>)` and require the splat.
    calls = [
        n for n in body
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Trainer"
    ]
    assert len(calls) == 1, f"expected exactly one Trainer(...) call, found {len(calls)}"
    splatted = [
        k.value.id for k in calls[0].keywords
        if k.arg is None and isinstance(k.value, ast.Name)
    ]
    assert ctor_name in splatted, (
        f"Trainer() no longer splats {ctor_name!r}; the config-derived kwargs "
        f"are not reaching the constructor"
    )
    use_line = calls[0].lineno

  # 3. Nothing may touch it in between.
    for node in body:
        line = getattr(node, "lineno", None)
        if line is None or not (bind_line < line < use_line):
            continue
        if isinstance(node, ast.Assign):
            for tgt in ast.walk(node):
                if isinstance(tgt, ast.Name) and tgt.id == ctor_name and isinstance(
                    tgt.ctx, ast.Store,
                ):
                    raise AssertionError(f"line {line}: {ctor_name} rebound before Trainer()")
                if isinstance(tgt, ast.Subscript) and getattr(
                    tgt.value, "id", None,
                ) == ctor_name and isinstance(tgt.ctx, ast.Store):
                    raise AssertionError(f"line {line}: {ctor_name}[...] assigned before Trainer()")
        if isinstance(node, ast.Delete):
            for tgt in node.targets:
                if getattr(getattr(tgt, "value", None), "id", None) == ctor_name:
                    raise AssertionError(f"line {line}: del from {ctor_name} before Trainer()")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and getattr(node.func.value, "id", None) == ctor_name
            and node.func.attr in {"pop", "clear", "update", "setdefault", "popitem"}
        ):
            raise AssertionError(
                f"line {line}: {ctor_name}.{node.func.attr}(...) mutates the "
                f"config-derived kwargs before they reach Trainer()"
            )


# ── The startup log line: the ONLY artifact naming the realized temperature ──
#
# ⚑ Ledger F2 (PR #373 independent review): the entry pre-declares UNRESOLVED as
# the expected outcome, and in this repo a null with no in-effect proof is void.
# Nothing a running trial emitted named the realized `policy_target_temp`:
# `params.json` is the LAUNCH config, and `scripts/audit_targets.py`'s
# "production training target" row is rebuilt from the flat `temperature` key --
# the selfplay SAMPLING temperature -- so it reads identically in both arms.
# `Trainer.__init__` now prints the value it actually installed.
#
# These tests are written to KILL a hard-coded line. A guard that cannot fire is
# worse than none, so a mutant printing any constant must turn at least one of
# them red, and the null control (deleting the line entirely) must too.


def _startup_temp_line(capsys: pytest.CaptureFixture[str]) -> str:
    lines = [
        ln for ln in capsys.readouterr().out.splitlines()
        if ln.startswith("[trainer] policy_target_temp=")
    ]
    assert len(lines) == 1, (
        "Trainer.__init__ must emit exactly one realized-temperature line; got "
        f"{len(lines)}. Deleting it makes the arm's in-effect proof unobservable."
    )
    return lines[0]


@pytest.mark.parametrize("temp", [1.0, 1.2, 1.5, 2.2, 0.75])
def test_the_startup_line_reports_the_REALIZED_temperature_not_a_constant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], temp: float,
) -> None:
    """⚑ THE MUTANT THIS EXISTS FOR: printing a literal (`policy_target_temp=1`,
    or the yaml's value re-read instead of `self.policy_target_temp`). The value
    is parsed back out of the emitted text and compared to the attribute that
    `_loss_kwargs` actually hands to `compute_loss`, over five temperatures, so
    NO single constant can satisfy the parametrization.
    """
    t = _trainer(tmp_path, policy_target_temp=temp)
    line = _startup_temp_line(capsys)

    printed = float(line.split("policy_target_temp=")[1].split()[0])
    assert printed == pytest.approx(temp), (
        f"the startup line says {printed} but the Trainer installed "
        f"{t.policy_target_temp} -- the operator cannot tell which arm ran"
    )
    assert printed == pytest.approx(t._loss_kwargs["policy_target_temp"]), (
        "the line must report the value that reaches compute_loss, not a "
        "separately-derived one"
    )


@pytest.mark.parametrize(("temp", "active"), [(1.0, False), (1.2, True), (0.75, True)])
def test_the_startup_line_reports_reshape_active_from_the_SHARED_predicate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], temp: float, active: bool,
) -> None:
    """`reshape_active` must agree with the predicate `retemper_main_policy_target`
    itself gates on -- a guard has to share the criterion's instrument. Asserted
    against the OBSERVED early return (identity of the returned object), not
    against a re-statement of `!= 1.0`, so a mutant that changes one and not the
    other is caught.
    """
    _trainer(tmp_path, policy_target_temp=temp)
    line = _startup_temp_line(capsys)
    printed = line.split("reshape_active=")[1].split()[0]
    assert printed == str(active)

    probe = _target([[0.7, 0.2, 0.07, 0.03]])
    observed_active = retemper_main_policy_target(probe, temp=temp) is not probe
    assert observed_active is active, "fixture disagrees with the real early return"
    assert printed == str(observed_active), (
        "the log line's reshape_active disagrees with whether the transform "
        "actually ran -- the claim and the arithmetic have drifted apart"
    )


def test_the_startup_line_fires_for_the_CONTROL_arm_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A line emitted only at `temp != 1.0` would make the control arm's log
    indistinguishable from a build that predates the knob, so `1.0` observed in
    the log would prove nothing. The default must print too."""
    _trainer(tmp_path)
    line = _startup_temp_line(capsys)
    assert "policy_target_temp=1.0 " in line
    assert "reshape_active=False" in line


@pytest.mark.parametrize("temp", [1.0, 1.3, 2.2])
def test_the_startup_line_READS_the_eval_pin_rather_than_asserting_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], temp: float,
) -> None:
    """⚑ REVIEW FINDING F5. `eval_pinned_temp=1` used to be a string LITERAL in
    the f-string -- a claim, not a measurement, in the one line whose entire
    justification is that it reports realized values. Under a mutant that
    un-pins the eval ruler (`_eval_loss_kwargs` returning `self.policy_target_temp`)
    the literal kept asserting the pin while the ruler moved with the arm.

    Now it is a read of `_eval_loss_kwargs`, which is only constructible at the
    END of `__init__` -- hence the print's placement. Assert the printed field
    against the property, at temperatures where a literal `1` and the real read
    would diverge if the pin ever broke.
    """
    t = _trainer(tmp_path, policy_target_temp=temp)
    line = _startup_temp_line(capsys)
    printed = float(line.split("eval_pinned_temp=")[1].split()[0])
    assert printed == pytest.approx(t._eval_loss_kwargs["policy_target_temp"]), (
        "eval_pinned_temp is not a read of _eval_loss_kwargs -- it is a claim"
    )
    assert printed == 1.0, "the eval ruler is no longer pinned"


class _UnpinnedEvalTrainer(Trainer):
    """A Trainer whose eval ruler is NOT pinned to 1.0.

    The only fixture that can tell a real read of ``_eval_loss_kwargs`` apart
    from the string literal the field used to be: on the production class the
    pin IS 1.0, so a literal `1.0` agrees with the property by coincidence and
    every assertion comparing the two passes. Here the property returns
    something else, so the printed field must move with it.
    """

    @property
    def _eval_loss_kwargs(self) -> dict[str, Any]:
        return {**self._loss_kwargs, "policy_target_temp": 7.25}


def test_eval_pinned_temp_is_a_READ_and_not_a_LITERAL(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ THE MUTANT THIS EXISTS FOR: `f"eval_pinned_temp=1.0"`. It survives every
    test that compares the printed field to `_eval_loss_kwargs`, because on the
    real class both are 1.0 -- the assertion checks the literal against itself.
    Subclassing moves the property away from 1.0, and only a genuine read
    follows it."""
    _UnpinnedEvalTrainer(
        _TinyPolicyModel(), device="cpu", lr=1e-3, optimizer="adamw",
        use_amp=False, log_dir=tmp_path, tb_log_interval=1000,
        prefetch_batches=False, policy_target_temp=1.3,
    )
    line = _startup_temp_line(capsys)
    printed = float(line.split("eval_pinned_temp=")[1].split()[0])
    assert printed == pytest.approx(7.25), (
        "eval_pinned_temp did not follow _eval_loss_kwargs -- it is a hard-coded "
        f"claim, not a measurement (line: {line!r})"
    )


def test_the_startup_line_survives_a_full_round_trip_of_the_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ REVIEW FINDING F7. `.6g` rendered `1.0000001` as `1` -- self-contradictory
    beside `reshape_active=True`, and byte-identical to the control arm's field.
    `!r` is shortest-round-trip, so the text must reconstruct the exact float."""
    temp = 1.0000001
    t = _trainer(tmp_path, policy_target_temp=temp)
    line = _startup_temp_line(capsys)
    printed_text = line.split("policy_target_temp=")[1].split()[0]
    assert float(printed_text) == t.policy_target_temp
    assert printed_text != "1", (
        "the arm's value renders identically to the control's -- an operator "
        "reading this line cannot tell the two arms apart"
    )
    assert "reshape_active=True" in line


# ── The shared predicate, pinned ON ITS OWN ─────────────────────────────────
#
# ⚑ REVIEW FINDING F6, and it is the cost of sharing an instrument. Every test
# above reads `reshape_active` and compares it to the OBSERVED early return --
# but both sides call `policy_target_temp_active`, so a change to the
# predicate's BOUNDARY moves them together and the comparison is satisfied by
# construction. The reviewer's mutant
#
#     return float(temp) != 1.0   ->   return abs(float(temp) - 1.0) > 1e-3
#
# passed the whole file. Sharing the definition was right; it just means the
# definition needs coverage that does not go through either consumer.


@pytest.mark.parametrize(
    ("temp", "expected"),
    [
        (1.0, False),          # the ONLY value that is off
        (1.0000001, True),     # inside a 1e-3 dead-band: kills that mutant
        (1.0001, True),        # inside a 1e-3 dead-band: kills that mutant
        (0.9999, True),        # inside it from below
        (1.001, True),         # the dead-band's own edge
        (0.5, True),           # the accepted floor
        (1.2, True), (1.5, True), (2.2, True),
        (float("nan"), True),  # invalid, but "does it bite" is still yes
        (float("inf"), True),
    ],
)
def test_policy_target_temp_active_is_exact_at_1_with_NO_dead_band(
    temp: float, expected: bool,
) -> None:
    """1.0 and only 1.0 is off. A tolerance band here would silently train a
    near-1.0 arm as the control while the log line reported it as armed."""
    assert policy_target_temp_active(temp) is expected


@pytest.mark.parametrize("temp", [1.0000001, 1.0001, 0.9999, 1.001])
def test_a_near_one_temperature_REALLY_reshapes_the_target(temp: float) -> None:
    """The predicate's answer has to match the arithmetic at the same boundary,
    checked WITHOUT going through the predicate: a fresh tensor object back from
    `retemper_main_policy_target` is the observable that the early return did
    not fire."""
    probe = _target([[0.7, 0.2, 0.07, 0.03]])
    out = retemper_main_policy_target(probe, temp=temp)
    assert out is not probe, f"temp={temp} took the identity early return"
    assert not torch.equal(out, probe), f"temp={temp} left the target unchanged"


def test_policy_target_temp_active_is_the_predicate_the_early_return_USES(
) -> None:
    """The two must not merely agree on today's values -- the early return has to
    be the predicate's negation for every probe, including the boundary ones the
    parametrisations above pin."""
    probe = _target([[0.6, 0.25, 0.1, 0.05]])
    for temp in (0.5, 0.9999, 1.0, 1.0000001, 1.001, 1.2, 2.2):
        took_identity = retemper_main_policy_target(probe, temp=temp) is probe
        assert took_identity is (not policy_target_temp_active(temp)), (
            f"temp={temp}: the early return and the predicate disagree"
        )
