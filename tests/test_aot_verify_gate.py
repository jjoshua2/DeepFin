"""Tests for the self-calibrating AOT verify gate.

Six properties carry this gate, and they are DIFFERENT properties. A previous
revision tested only the first and shipped two live holes:

0. **⚑⚑ THE FLOOR IS THE PHYSICAL bf16 ERROR** — one ULP of the RAW logit,
   ``2^(floor(log2|z|) - 7)``. Not a relative nudge of a centred row. The
   centred form was 12-22x too small on the WDL head and 0.3x on the policy
   head, because the two heads have very different offset/spread ratios, so a
   package that was EXACTLY one bf16 ULP off — the best a package can
   physically be — read ``wdl_mean=x12.25, wdl_rows_over=26/64, FAIL``. The
   correct property is shift-COVARIANCE: the offset is what the net emits, and
   bf16 error scales with it. Properties 1-5 all held while that was true.
   ⚑ The suite ACTIVELY ENFORCED the defect before this was found — a mutant
   that removed the centring was killed by three tests. A green suite is not
   evidence the physics is right; it is evidence the suite agrees with itself.

1. **Sharpness invariance** — the verdict must not move when the net's top-1
   grows. That is the exact 2026-08-15 failure, where identical deployed
   packages went PASS -> FAIL because the policy concentrated 4.4x.
2. **Scale-freedom** — the verdict must depend on the RATIO, never on absolute
   TV. Sharpness invariance alone leaves a frozen denominator (`tv / 0.02`)
   alive, because that scenario holds aot_off == ctl_off at a single noise level.
3. **Non-degeneracy** — the floor must never be zero. The batch-shape control
   is bitwise-identical on CPU from n=2 to n=128, and an epsilon clamp turns
   "no information" into "x399, FAIL" on a healthy package.
4. **⚑⚑ PADDING INVARIANCE** — the floor must not grow with the illegal-move
   mask. This one was added on 2026-08-15 after the gate was found
   ARITHMETICALLY INCAPABLE OF FAILING at the production policy width: the
   4672-slot vector is 60% -1e9 sentinel, the whole-row mean sat near -6.0e8,
   and the floor saturated at 0.9719 against 0.0189 on the native 1858-wide
   array. Since a row TV is bounded by 1, no policy arm could exceed its bound —
   a completely wrong policy verified clean at pol_mean x1.17, argmax 0/256.
   Properties 1-3 all held while that was true; every one of them was measured
   on a narrow array. **A property proven at the wrong SHAPE is not proven.**
5. **The instrument checks itself** — the two floor estimates cross-check, and a
   >5x divergence FAILS the bucket rather than warning. A broken floor is the
   one failure a ratio gate cannot self-detect, and it errs in the PASSING
   direction. Property 4's defect would have shown here as ~45x.

Plus the standing rule for this repo: every knob must be shown to REACH the
comparison, and every branch must be shown to be able to FAIL. A gate that
cannot fail is the defect this codebase produces most often — and `ok` is an OR
over five arms, so several tests below parse the detail line to prove WHICH arm
fired. A test that only reads the verdict goes green when the arm it is named
after is deleted.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.build_aot_packages as MOD
from scripts.build_aot_packages import (
    _compare_bucket,
    bf16_ulp_perturbation,
    eager_batch_shape_control,
    mean_row_tv,
    row_tv,
    tail_row_tv,
)

# The production policy vector: `_policy_output_full` emits 4672 slots of which
# only 1858 carry a real logit; the rest hold a -1e9 sentinel that softmax sends
# to exactly 0. Every "production-shaped" test below builds that layout, because
# the gate's blocking defect was invisible at any other width.
LIVE_LOGITS = 1858
FULL_POLICY = 4672
SENTINEL = -1e9


def _probs(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sentinel_padded(live: np.ndarray) -> np.ndarray:
    """`live` (n, 1858) widened to the production (n, 4672) with -1e9 padding."""
    full = np.full((int(live.shape[0]), FULL_POLICY), SENTINEL, dtype=np.float32)
    full[:, :LIVE_LOGITS] = live
    return full


def _arms(detail: str) -> dict[str, float]:
    """Parse `_compare_bucket`'s detail line back into its individual arms.

    Several tests below have to prove WHICH arm fired, not merely that the
    bucket failed. `ok` is an OR over arms, so a verdict alone cannot
    distinguish "the arm under test caught it" from "some other arm did" — and
    a test that cannot tell those apart goes green when the arm it names is
    deleted.
    """
    out: dict[str, float] = {
        k: float(v) for k, v in re.findall(r"(\w+)=x(inf|[\d.]+)", detail)
    }
    out.update({
        k: float(v) for k, v in re.findall(r"(\w+_rows_over)=(\d+)", detail)
    })
    return out


def _live_mask(z: np.ndarray) -> np.ndarray:
    """The gate's live window, restated here so the healthy models below share
    no code with the function they are used to calibrate."""
    return z > (np.max(z, axis=-1, keepdims=True) - np.float32(80.0))


def _independent_abs_ulp(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A ±1 raw bf16 ULP jitter, written INDEPENDENTLY of the gate.

    ⚑⚑ THIS IS THE POINT OF F2. The healthy numerator must not be produced by
    the denominator's own function at another seed: that measures
    self-consistency, and it reads 1.0 whatever the floor computes — including
    when the floor computes something physically meaningless. So the physics is
    restated from scratch here (via ``log2``/``exp2``, where the gate uses
    ``frexp``): bf16 keeps 8 significand bits, so its spacing at ``|z|`` is
    ``2^(floor(log2|z|) - 7)``.
    """
    z = z.astype(np.float32)
    mag = np.abs(z)
    exponent = np.floor(np.log2(mag, out=np.full_like(mag, -1e30), where=mag > 0))
    step = np.where(mag > 0, np.exp2(exponent - 7.0), 0.0).astype(np.float32)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=z.shape)
    return np.where(_live_mask(z), z + sign * step, z).astype(np.float32)


def _independent_rel_jitter(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A gaussian jitter at bf16's RELATIVE scale — a second, different healthy
    model, so no conclusion rests on one choice of "healthy". It is ~0.6x an
    absolute ULP in TV, so it is the friendlier of the two and is never used to
    justify a bound."""
    z = z.astype(np.float32)
    jittered = (z * (1.0 + rng.normal(0, 2.0 ** -8, size=z.shape))).astype(np.float32)
    return np.where(_live_mask(z), jittered, z).astype(np.float32)


def _gate(a_p, a_w, r_p, r_w, c_p=None, c_w=None, *, tv_ratio_max=2.0):
    return _compare_bucket(
        aot_pol=a_p, aot_wdl=a_w, ref_pol=r_p, ref_wdl=r_w,
        ctl_pol=c_p, ctl_wdl=c_w, tv_ratio_max=tv_ratio_max,
    )


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def test_mean_row_tv_bounds() -> None:
    p = np.array([[0.5, 0.5], [0.25, 0.75]])
    assert mean_row_tv(p, p) == pytest.approx(0.0)
    assert mean_row_tv(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])) == pytest.approx(1.0)


def test_mean_row_tv_averages_rows_not_maxes_them() -> None:
    """A max would report 1.0 here; a mean reports 0.5. That is half the fix."""
    a = np.array([[1.0, 0.0], [0.5, 0.5]])
    b = np.array([[0.0, 1.0], [0.5, 0.5]])
    assert mean_row_tv(a, b) == pytest.approx(0.5)


def test_tail_row_tv_sees_what_the_mean_dilutes() -> None:
    """The other half: a mean alone has no power against row-local damage."""
    n = 1000
    a = np.tile(np.array([0.5, 0.5]), (n, 1))
    b = a.copy()
    b[:20] = np.array([1.0, 0.0])  # 2% of rows maximally wrong
    assert mean_row_tv(a, b) == pytest.approx(0.01, abs=1e-9)  # ~invisible
    assert tail_row_tv(a, b, quantile=0.99) > 0.4  # loud


def test_tail_row_tv_is_a_quantile_not_an_extreme_value() -> None:
    """It must estimate the same population quantity at every bucket size.

    A max grows with N and is what made the old gate a lottery; a quantile does
    not. Same distribution, 16x the rows, same answer.
    """
    rng = np.random.default_rng(0)
    vals = []
    for n in (256, 4096):
        ref = rng.normal(0, 4.0, size=(n, 64))
        aot = ref + rng.normal(0, 0.05, size=ref.shape)
        vals.append(tail_row_tv(_probs(aot), _probs(ref)))
    assert vals[1] == pytest.approx(vals[0], rel=0.15), (
        f"tail statistic moved with bucket size: {vals}"
    )


def test_row_tv_is_per_row() -> None:
    a = np.array([[1.0, 0.0], [0.5, 0.5]])
    b = np.array([[0.0, 1.0], [0.5, 0.5]])
    assert row_tv(a, b).tolist() == pytest.approx([1.0, 0.0])


# --------------------------------------------------------------------------
# the floor
# --------------------------------------------------------------------------

def _ulp_scale_control(ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A batch-shape control at the bf16 floor's OWN scale, on any head.

    ⚑ An ABSOLUTE noise level cannot do that job any more. The floor is one raw
    bf16 ULP, so it scales with |z| — one sd that lands near the floor on
    128-wide sd-4 policy logits sits 7.8x above it on N(0,1) WDL, and the bucket
    then fails as FLOOR-DIVERGENCE for a reason the test using it is not about.
    A RELATIVE jitter at 2^-8 is within 2x of the floor on both heads at every
    logit magnitude, so it is the control shape these fixtures want.
    """
    return (ref * (1.0 + rng.normal(0, 2.0 ** -8, size=ref.shape))).astype(np.float32)


def test_the_perturbation_is_ONE_RAW_BF16_ULP_of_each_logit() -> None:
    """⚑⚑ THE F1 CONTRACT, asserted on the LOGITS rather than on a statistic.

    bf16 rounding error is one ULP of the RAW value — ``2^(floor(log2|z|) - 7)``,
    because bf16 carries 8 significand bits. An earlier revision centred each
    row on its live mean and applied a RELATIVE 2^-8, i.e. ``|z - rowmean| *
    2^-8``, on the argument that the raw logits' common offset was "arbitrary".
    It is not arbitrary: it is what the net emits, and a net whose head bias
    moved emits larger-magnitude logits whose bf16 rounding error genuinely is
    larger. The physical property is shift-COVARIANCE.

    The two disagree by exactly the offset/spread ratio, which is why the defect
    was head-dependent: at the production WDL regime (row mean -8.60, within-row
    spread 1.00) the raw ULP of -8.60 is 0.0625 while ``|z - mean| * 2^-8`` is
    0.0039 — **16x too small**. This test pins the raw form entry by entry and
    states the centred number it must NOT produce.
    """
    z = np.array([[-8.60, -15.5625, 0.5, 3.0, SENTINEL]], dtype=np.float32)
    out = bf16_ulp_perturbation(z, seed=0)
    delta = np.abs(out - z)[0]

    # 2^(floor(log2|z|) - 7), by hand: |−8.60| and 15.5625 are in [8,16) -> 2^-4;
    # 0.5 is in [0.5,1) -> 2^-8; 3.0 is in [2,4) -> 2^-6.
    expected = np.array([0.0625, 0.0625, 2.0 ** -8, 2.0 ** -6], dtype=np.float32)
    assert np.array_equal(delta[:4], expected), (
        f"perturbation is not one raw bf16 ULP: {delta[:4]} != {expected}"
    )
    assert delta[4] == 0.0, "the -1e9 sentinel must be left exactly where it is"

    # ⚑ And say what the reverted form would give, ON THE REAL WDL GEOMETRY —
    # row mean -8.60, within-row spread 1.00 — because that offset/spread ratio
    # IS the defect. A row with a spread comparable to its offset does not
    # separate the two forms and would pin neither.
    wdl_like = np.array([[-8.60, -9.60, -7.60]], dtype=np.float32)
    wdl_delta = np.abs(bf16_ulp_perturbation(wdl_like, seed=0) - wdl_like)[0]
    centred = np.abs(wdl_like[0] - wdl_like[0].mean()) * (2.0 ** -8)
    assert wdl_delta[0] == np.float32(0.0625), wdl_delta
    assert wdl_delta[0] / centred[1] > 10.0, (
        f"raw ULP {wdl_delta[0]:.5f} vs centred-relative {centred[1]:.5f} — the "
        "scenario does not separate the two forms, so it cannot pin either"
    )
    assert centred[0] == 0.0, (
        "the centre entry's centred-relative nudge is EXACTLY zero while its "
        "true bf16 error is 0.0625 — the clearest statement of why the centred "
        "form was not a bf16 ULP"
    )


def test_ulp_floor_is_never_zero_where_the_shape_control_is() -> None:
    """⚑ THE F2 regression test.

    Round-tripping already-bf16 logits through bf16 is a no-op, and the
    batch-shape control is bitwise identical on CPU — both give a floor of
    exactly 0. A +/-1 ULP nudge of a WIDE row does not.

    ⚑ "Never zero" is a claim about a wide row, not a universal one: see
    `test_three_equal_wdl_logits_can_still_give_a_zero_floor`, where a real
    3-wide WDL row does produce exactly 0 at some floor seeds.
    """
    rng = np.random.default_rng(0)
    ref = torch.from_numpy(
        rng.normal(0, 6.0, size=(64, 256)).astype(np.float32)
    ).to(torch.bfloat16).float().numpy()

    round_tripped = torch.from_numpy(ref).to(torch.bfloat16).float().numpy()
    assert np.array_equal(round_tripped, ref), "premise: ref logits are already bf16"
    assert mean_row_tv(_probs(round_tripped), _probs(ref)) == 0.0

    nudged = bf16_ulp_perturbation(ref, seed=0)
    assert mean_row_tv(_probs(nudged), _probs(ref)) > 0.0


def test_ulp_floor_tracks_sharpness() -> None:
    """It is a FLOOR only if it grows as the softmax concentrates."""
    rng = np.random.default_rng(1)
    floors = []
    for scale in (1.0, 6.0):
        ref = (rng.normal(0, 1.0, size=(256, 512)) * scale).astype(np.float32)
        floors.append(mean_row_tv(_probs(bf16_ulp_perturbation(ref, seed=0)), _probs(ref)))
    assert floors[1] > 3 * floors[0], f"floor did not track sharpness: {floors}"


def test_a_degenerate_shape_control_does_not_fail_a_healthy_package() -> None:
    """⚑ THE measured F2 failure: TV 4e-4 vs control 0 read x399 FAIL.

    ctl == ref exactly is the CPU regime. The gate must fall through to the ULP
    floor and PASS, not divide by an epsilon.
    """
    rng = np.random.default_rng(3)
    ref = (rng.normal(0, 6.0, size=(64, 256)) ).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(64, 3)).astype(np.float32)
    aot = ref + rng.normal(0, 0.002, size=ref.shape).astype(np.float32)
    aot_w = ref_w + rng.normal(0, 0.002, size=ref_w.shape).astype(np.float32)

    ok, detail, _, _ = _gate(aot, aot_w, ref, ref_w, ref.copy(), ref_w.copy())
    assert ok, f"healthy package failed against a degenerate control: {detail}"


def test_the_floor_takes_the_larger_of_the_two_estimates() -> None:
    """A large empirical control must WIDEN the gate relative to ULP alone.

    ⚑ The widening window is now BOUNDED, and that is deliberate. Since
    `_FLOOR_DIVERGENCE_MAX = 5.0`, a control more than 5x the ULP estimate no
    longer widens anything — it FAILS the bucket as a broken instrument. So the
    scenario has to live inside that window: RE-MEASURED after the raw-ULP fix,
    ULP-only reads pol_mean x2.65 (FAIL at the 2.0 default) and the matched
    control pulls it to x1.05 (PASS). The pre-guard version of this test used a
    0.30 control, which fails for the OTHER reason — a test that would have gone
    green on a floor that never widened.

    ⚑ The noise level moved 0.12 -> 0.16 with the F1 fix, and that is the fix
    showing up rather than a tuned test: a raw ULP is on average ~1.5x a
    centred-relative 2^-8 nudge on zero-mean logits, so the same package sits
    lower against the honest floor. At 0.12 this now reads x1.99 — under the
    bound, and the test would have asserted nothing.

    The WDL side of this scenario is deliberately bit-identical (aot_w == ref_w
    == ctl_w), which also exercises the degenerate-control label: the detail
    line must report `wdl:ulp-only(ctl-degenerate)` rather than pretending a
    non-measurement contributed to the floor.
    """
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 4.0, size=(128, 256)).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(128, 3)).astype(np.float32)
    aot = ref + rng.normal(0, 0.16, size=ref.shape).astype(np.float32)
    aot_w = ref_w.copy()

    tight, tight_detail, _, _ = _gate(aot, aot_w, ref, ref_w)  # ULP floor only
    ctl = ref + rng.normal(0, 0.16, size=ref.shape).astype(np.float32)
    loose, loose_detail, _, _ = _gate(aot, aot_w, ref, ref_w, ctl, ref_w.copy())
    assert not tight, "the ULP-only floor should have been too tight here"
    assert _arms(tight_detail)["pol_mean"] > 2.0, (
        "the ULP-only verdict must come from the RATIO arm, or this test says "
        f"nothing about the floor: {tight_detail}"
    )
    assert loose, (
        "an empirical control at the SAME noise level as the package must make "
        f"the gate pass; it did not widen the floor: {loose_detail}"
    )
    assert "FLOOR-DIVERGENCE" not in loose_detail, loose_detail
    assert "floor=pol:shape+ulp wdl:ulp-only(ctl-degenerate)" in loose_detail, (
        "the floor SOURCE must be reported per head and must name a degenerate "
        f"control as one: {loose_detail}"
    )


def test_both_floors_zero_demands_exactness() -> None:
    """Constant logits carry no scale. The only honest verdict is exact."""
    ref = np.zeros((8, 4), dtype=np.float32)
    ref_w = np.zeros((8, 3), dtype=np.float32)
    ok_same, _, _, _ = _gate(ref.copy(), ref_w.copy(), ref, ref_w, ref.copy(), ref_w.copy())
    assert ok_same, "identical outputs must pass even with a zero floor"

    bad = ref.copy()
    bad[0, 0] = 5.0
    ok_diff, detail, _, _ = _gate(bad, ref_w.copy(), ref, ref_w, ref.copy(), ref_w.copy())
    assert not ok_diff, f"a zero floor must not absorb a real difference: {detail}"


# --------------------------------------------------------------------------
# the three headline properties
# --------------------------------------------------------------------------

def _case(*, aot_off: float, ctl_off: float, n: int = 64, sharp: float = 1.0):
    rng = np.random.default_rng(5)
    ref = (rng.normal(0.0, 1.0, size=(n, 40)) * sharp).astype(np.float32)
    # bf16 error is RELATIVE: absolute logit error scales with logit magnitude.
    # Holding it fixed while scaling `ref` would model shrinking relative
    # precision, which is not what sharpening does.
    aot = ref + rng.normal(0.0, aot_off * sharp, size=ref.shape).astype(np.float32)
    ctl = ref + rng.normal(0.0, ctl_off * sharp, size=ref.shape).astype(np.float32)
    wdl_ref = (rng.normal(0.0, 1.0, size=(n, 3)) * sharp).astype(np.float32)
    return (
        aot, wdl_ref + rng.normal(0.0, aot_off * sharp, size=(n, 3)).astype(np.float32),
        ref, wdl_ref,
        ctl, wdl_ref + rng.normal(0.0, ctl_off * sharp, size=(n, 3)).astype(np.float32),
    )


def test_gate_passes_when_aot_matches_the_control_level() -> None:
    """⚑ 0.004, not 0.02: bf16's relative ULP is 2^-8 = 0.0039, so on N(0,1)
    logits a HEALTHY package sits near 0.004 absolute. A 0.02 control is ~5
    ULPs, and the floor cross-check now (correctly) calls that out: measured
    ulp=0.00066 vs shape=0.00530, x8.1 > 5, FLOOR-DIVERGENCE. Passing a package
    against an instrument that disagrees with itself 8x is the verdict-off-a-
    failed-stage error, so the SCENARIO was wrong, not the gate.
    """
    ok, detail, _, _ = _gate(*_case(aot_off=0.004, ctl_off=0.004))
    assert ok, detail


def test_gate_fails_when_aot_is_much_worse_than_the_control() -> None:
    # ctl_off is at the bf16 ULP scale so the control agrees with the analytic
    # floor; a 0.02 control trips FLOOR-DIVERGENCE and this test would then pass
    # without any arm having judged the package at all.
    ok, detail, _, _ = _gate(*_case(aot_off=0.60, ctl_off=0.004))
    assert not ok, detail
    assert "FLOOR-DIVERGENCE" not in detail, (
        f"the failure must come from the package, not from a broken floor: {detail}"
    )
    assert _arms(detail)["pol_mean"] > 2.0, detail


def test_gate_verdict_is_invariant_to_policy_sharpness() -> None:
    """⚑ Property 1 — THE regression test for the 2026-08-15 failure."""
    verdicts, old_gate_maxdev = [], []
    for sharp in (1.0, 6.0):
        a_p, a_w, r_p, r_w, c_p, c_w = _case(aot_off=0.004, ctl_off=0.004, sharp=sharp)
        ok, detail, _, _ = _gate(a_p, a_w, r_p, r_w, c_p, c_w)
        verdicts.append(ok)
        old_gate_maxdev.append(float(np.max(np.abs(_probs(a_p) - _probs(r_p)))))
        # ⚑ EQUALITY OF VERDICTS IS SATISFIED BY (False, False). At the old
        # 0.02 noise level both readings became FLOOR-DIVERGENCE failures, so
        # this test kept passing while measuring nothing. Pin the level.
        assert ok, f"sharp={sharp} is not the healthy regime this test needs: {detail}"

    assert verdicts[0] == verdicts[1], f"verdict moved with sharpness alone: {verdicts}"
    # And confirm the OLD statistic really would have moved — otherwise this
    # test proves nothing about the defect it claims to guard.
    assert old_gate_maxdev[1] > 3 * old_gate_maxdev[0], (
        f"sharpening did not move the old max-dev statistic ({old_gate_maxdev}); "
        "the scenario does not reproduce the defect"
    )


def test_gate_is_scale_free_in_the_noise_floor() -> None:
    """⚑ Property 2 — kills a frozen denominator, which property 1 does not.

    ⚑ THE TESTABLE WINDOW IS NOW CAPPED AT ~5x, by `_FLOOR_DIVERGENCE_MAX`.
    Moving the empirical noise level while holding `ref` fixed moves the
    batch-shape floor and leaves the analytic ULP floor where it is, so the two
    estimates separate by exactly the factor this test wants to sweep. Measured
    on this scenario, the policy/WDL divergence runs 2.61/1.24 at off=0.002 and
    1.92/4.02 at off=0.010; by off=0.014 the WDL head reads 5.64 and the bucket
    fails as a broken instrument rather than as a broken package. So the sweep
    is 0.002 -> 0.010 (a 5.0x move in absolute TV, asserted at >4x) instead of
    the pre-guard 0.02 -> 0.5, and that is a real, intended narrowing: the gate
    no longer claims to be scale-free across a floor its own cross-check says
    cannot be right.
    """
    verdicts, tvs, details = [], [], []
    for off in (0.002, 0.010):
        a_p, a_w, r_p, r_w, c_p, c_w = _case(aot_off=off, ctl_off=off)
        ok, detail, _, _ = _gate(a_p, a_w, r_p, r_w, c_p, c_w)
        verdicts.append(ok)
        details.append(detail)
        tvs.append(mean_row_tv(_probs(a_p), _probs(r_p)))
    assert all(verdicts), (
        f"a scale-free gate must pass both noise levels: {verdicts} at TV {tvs}; {details}"
    )
    assert tvs[1] > 4 * tvs[0], f"noise levels too close ({tvs}) to separate the designs"


def test_row_local_damage_fails_even_though_the_mean_is_clean() -> None:
    """⚑ The coverage the old whole-array max had, that a mean alone loses.

    A kernel wrong on 2% of rows — a boundary/indexing defect at one bucket
    size — barely moves the mean. The tail arm is what catches it.
    """
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 4.0, size=(500, 128)).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(500, 3)).astype(np.float32)
    ctl = _ulp_scale_control(ref, rng)
    ctl_w = _ulp_scale_control(ref_w, rng)

    aot = _ulp_scale_control(ref, rng)
    aot[:10] = rng.normal(0, 4.0, size=(10, 128))  # 2% of rows: garbage

    ok, detail, _, _ = _gate(aot, ctl_w.copy(), ref, ref_w, ctl, ctl_w)
    assert not ok, f"row-local corruption passed the gate: {detail}"
    assert "FLOOR-DIVERGENCE" not in detail, detail


# --------------------------------------------------------------------------
# no branch may be unable to fail
# --------------------------------------------------------------------------

def test_a_missing_control_still_gates_and_still_says_so() -> None:
    """⚑ With ctl=None there is NO degraded pass-through: the ULP floor gates.

    A previous revision returned `ok = argmax_rate >= argmax_min` here, so
    replacing the whole branch with `ok = True` was invisible to the suite.
    """
    # ⚑ bf16's relative ULP is 2^-8 = 0.0039, so on N(0,1) logits a HEALTHY
    # package sits near 0.004 absolute -- not 0.02, which is ~5 ULPs and which
    # the ULP-only floor correctly rejects. Using 0.02 here would have been a
    # scenario that mismodels the deployment regime, not a gate defect.
    a_p, a_w, r_p, r_w, _, _ = _case(aot_off=0.002, ctl_off=0.002)
    ok, detail, _, _ = _gate(a_p, a_w, r_p, r_w, None, None)
    assert ok, detail
    assert "ulp-only" in detail, "the weaker floor must announce itself"

    bad_p, bad_w, r_p, r_w, _, _ = _case(aot_off=1.5, ctl_off=0.02)
    ok_bad, detail_bad, _, _ = _gate(bad_p, bad_w, r_p, r_w, None, None)
    assert not ok_bad, f"the no-control branch cannot fail: {detail_bad}"


def test_compare_bucket_reports_argmax_but_does_not_gate_on_it() -> None:
    """Argmax is pooled by the caller; per-trial it is a coin flip at small n."""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(128, 40)).astype(np.float32)
    garbage = rng.normal(size=(128, 40)).astype(np.float32)
    _, _, matches, rows = _gate(garbage, ref[:, :3], ref, ref[:, :3], ref, ref[:, :3])
    assert rows == 128
    assert matches < 20, "independent logits should agree on argmax ~1/40 of the time"


def test_tv_ratio_max_changes_the_verdict() -> None:
    """The knob must REACH the comparison, not merely be accepted.

    ⚑ The damage has to sit between the ratio bound and the row-exceedance
    bound. `tv_ratio_max` governs the three ratio arms ONLY; the count arms are
    gated at zero in units of the package's own floor and do not move with it
    (see `test_the_row_exceedance_arm_is_not_loosened_by_tv_ratio_max`). The
    pre-respec version used aot_off=0.30, which drives every row past
    `_ROW_EXCEED_K` — so `tv_ratio_max=1e9` could not make it pass and this test
    could not tell a live knob from a dead one. Measured here: x2.89/2.49/3.80
    on the three arms with zero exceeding rows.
    """
    args = _case(aot_off=0.015, ctl_off=0.004)
    tight_ok, tight_detail, _, _ = _gate(*args, tv_ratio_max=1.5)
    wide_ok, wide_detail, _, _ = _gate(*args, tv_ratio_max=1e9)
    assert not tight_ok, tight_detail
    assert wide_ok, wide_detail
    assert _arms(tight_detail)["pol_rows_over"] == 0, tight_detail
    assert _arms(tight_detail)["wdl_rows_over"] == 0, tight_detail


# --------------------------------------------------------------------------
# the empirical control
# --------------------------------------------------------------------------

def test_eager_control_returns_none_when_batch_cannot_be_rechunked() -> None:
    pol, wdl = eager_batch_shape_control(torch.nn.Identity(), torch.zeros(1, 3))
    assert pol is None
    assert wdl is None


class _RowIdentity(torch.nn.Module):
    """Emits a row-dependent output, so a misordered control is detectable."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int] = []

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen.append(int(x.shape[0]))
        tag = x[:, :1]
        return {
            "policy": tag.expand(-1, 1858).contiguous(),
            "wdl": tag.expand(-1, 3).contiguous(),
        }


def test_eager_control_uses_a_different_batch_shape() -> None:
    m = _RowIdentity()
    eager_batch_shape_control(m, torch.zeros(16, 3))
    assert m.seen, "control never ran the model"
    assert all(s != 16 for s in m.seen), f"control re-ran at the SAME shape: {m.seen}"
    assert sum(m.seen) == 16, f"control did not cover every row exactly once: {m.seen}"


def test_eager_control_preserves_row_ORDER() -> None:
    """⚑ A misaligned control inflates the floor and makes the gate permissive.

    Shapes and counts can both be right while the rows are scrambled, so
    asserting `sum(chunks) == n` does not cover this.
    """
    x = torch.arange(16, dtype=torch.float32).reshape(16, 1).repeat(1, 3)
    pol, _ = eager_batch_shape_control(_RowIdentity(), x)
    assert pol is not None
    assert pol[:, 0].tolist() == pytest.approx(list(range(16))), (
        f"control returned rows out of order: {pol[:, 0].tolist()}"
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7, 8, 9, 15, 16, 64, 1190])
def test_eager_control_covers_every_row_exactly_once(n: int) -> None:
    m = _RowIdentity()
    pol, _ = eager_batch_shape_control(m, torch.zeros(n, 3))
    assert pol is not None
    assert pol.shape[0] == n
    assert sum(m.seen) == n, (n, m.seen)
    assert all(s != n for s in m.seen), (n, m.seen)


# --------------------------------------------------------------------------
# end-to-end: the knobs and the control must reach the real path
# --------------------------------------------------------------------------

class _StubModel(torch.nn.Module):
    input_extra_features = "v2_threats"
    input_history_encoding = "lc0_root"
    policy_output_format = "lc0_1858"

    def __init__(self, jitter: float = 0.0) -> None:
        super().__init__()
        self.jitter = float(jitter)
        # verify_packages reads the device off next(model.parameters())
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return _stub_logits(x)


def _stub_projection() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return (torch.randn(64, 1858, generator=g), torch.randn(64, 3, generator=g))


def _stub_logits(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """⚑ ROW-DETERMINISTIC, which is the whole point of this stub.

    An earlier version drew fresh `torch.randn(n, ...)` per call. That made the
    chunked eager control a COMPLETELY different draw, so its TV was ~maximal
    and the floor swamped every arm — the end-to-end tests then passed a broken
    package and could not tell the knobs apart. A stub whose output depends on
    the batch SHAPE rather than on the ROWS models a network that does not
    exist, and it silently disables the gate under test.

    A fixed projection of a fixed input slice is chunk-invariant, so the eager
    control degenerates to TV 0 exactly as it does on real CPU hardware, and the
    ULP floor is what carries the gate — the regime worth testing.
    """
    wp, ww = _stub_projection()
    feat = x.reshape(int(x.shape[0]), -1)[:, :64].float()
    return {"policy": (feat @ wp) * 0.25, "wdl": (feat @ ww) * 0.25}


class _StubPackage:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def __call__(self, xt: torch.Tensor) -> dict[str, torch.Tensor]:
        out = _stub_logits(xt)
        if self.offset:
            n = int(xt.shape[0])
            g2 = torch.Generator().manual_seed(99)
            out["policy"] = out["policy"] + torch.randn(n, 1858, generator=g2) * self.offset
            out["wdl"] = out["wdl"] + torch.randn(n, 3, generator=g2) * self.offset
        return out

    def load_constants(self, *_a: object, **_k: object) -> None:
        return None

    def get_constant_fqns(self) -> list[str]:
        # Non-empty: verify_packages refuses a package with no externalized
        # constants, and that refusal would mask every gate verdict below.
        return ["anchor"]


def _run_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, offset: float,
                tv_ratio_max: float = 1.5, verify_n: int = 1, argmax_min: float = 0.90):
    monkeypatch.setattr(MOD, "build_aot_constants", lambda *a, **k: {})
    monkeypatch.setattr(MOD, "_aoti_load_package", lambda p: _StubPackage(offset))
    (tmp_path / "chess_b8.pt2").write_bytes(b"x")
    return MOD.verify_packages(
        out_dir=tmp_path, model=_StubModel(), buckets=[8], max_batch=8,
        input_planes=175, tv_ratio_max=tv_ratio_max, verify_n=verify_n,
        argmax_min=argmax_min, seed=3,
    )


def test_end_to_end_a_healthy_package_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    n_pass, n_fail, rows = _run_verify(tmp_path, monkeypatch, offset=0.0)
    assert (n_pass, n_fail) == (1, 0), rows


def test_end_to_end_a_broken_package_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """⚑ Kills 'compute the control, then discard it at the call site'.

    Nothing above this line exercises verify_packages' own wiring; a mutant that
    passed ctl_pol=None wholesale left every unit test green.
    """
    n_pass, n_fail, rows = _run_verify(tmp_path, monkeypatch, offset=4.0)
    assert (n_pass, n_fail) == (0, 1), rows


def test_end_to_end_tv_ratio_max_reaches_the_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # argmax_min is disarmed so this isolates the TV arm: the two criteria are
    # independent, and a broken package trips BOTH.
    # ⚑ offset 0.005, not 4.0. `tv_ratio_max` bounds the three RATIO arms only;
    # at offset 4.0 every row also clears `_ROW_EXCEED_K`, which is gated at
    # zero and deliberately does NOT move with this knob, so `1e12` could not
    # produce a PASS and the test would report a live knob as dead. Measured at
    # 0.005: x1.67/1.52/3.30 with pol_rows_over=0 and wdl_rows_over=0.
    wide = _run_verify(tmp_path, monkeypatch, offset=0.005, argmax_min=0.0,
                       tv_ratio_max=1e12)
    tight = _run_verify(tmp_path, monkeypatch, offset=0.005, argmax_min=0.0,
                        tv_ratio_max=1.5)
    assert wide[0] == 1, wide
    assert tight[1] == 1, tight


def test_end_to_end_argmax_min_reaches_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run_verify(tmp_path, monkeypatch, offset=0.0, argmax_min=0.90)[0] == 1
    assert _run_verify(tmp_path, monkeypatch, offset=0.0, argmax_min=1.01)[1] == 1


def test_end_to_end_verify_n_changes_the_pooled_row_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⚑ verify_n was silently ignorable: nothing consumed its extra rows."""
    rows_1 = _run_verify(tmp_path, monkeypatch, offset=0.0, verify_n=1)[2]
    rows_3 = _run_verify(tmp_path, monkeypatch, offset=0.0, verify_n=3)[2]
    assert "pooled_argmax=8/8" in rows_1[0][2], rows_1
    assert "pooled_argmax=24/24" in rows_3[0][2], rows_3


def test_main_returns_nonzero_when_a_bucket_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A verify run that FAILS must not exit 0 — CI would never notice."""
    monkeypatch.setattr(MOD, "verify_packages", lambda **k: (0, 1, [(8, "FAIL", "x")]))
    # ⚑ *a as well as **k: build_reference_model is called POSITIONALLY. A
    # **k-only lambda raises, main's except returns 1, and `rc != 0` then holds
    # for the wrong reason — a vacuous pass that let "main always returns 0"
    # survive mutation.
    monkeypatch.setattr(MOD, "build_reference_model", lambda *a, **k: _StubModel())
    monkeypatch.setattr(MOD, "load_model_config", lambda p: _StubModel())
    monkeypatch.setattr(MOD, "load_checkpoint_state_dict", lambda p: {})
    monkeypatch.setattr(MOD, "_require_cuda", lambda action: None)
    rc = MOD.main([
        "--checkpoint", "x.pt", "--verify-only", "--out-dir", str(tmp_path),
        "--buckets", "8", "--max-batch", "8",
    ])
    assert rc != 0, "a failing verify exited 0"
    # And prove the verify branch was actually REACHED, so the exit code came
    # from the failed count rather than from an exception on the way there.
    assert "failed=1" in capsys.readouterr().out


def test_the_tail_arm_fires_where_the_mean_arm_does_NOT() -> None:
    """⚑ Otherwise the tail arm is decorative — deleting it stays green.

    A first attempt at a row-local test used damage severe enough that the MEAN
    arm caught it too, so removing the tail arm entirely left the suite passing.
    This scenario is tuned so the mean ratio stays UNDER the threshold and only
    the tail crosses it: 8% of rows perturbed moderately, which is exactly the
    signature of a boundary defect at one bucket size.

    ⚑ AND IT NOW HAS TO DODGE THE ROW-EXCEEDANCE ARM TOO. With the previous
    settings (1.5% of rows at sd 1.2) the damaged rows also cleared
    `_ROW_EXCEED_K * floor_tail`, so `pol_rows_over` read 22 and deleting the
    tail RATIO arm left this green — the same decorativeness the docstring above
    warns about, one arm later. Measured at the settings below: pol_mean x0.49,
    pol_tail x2.38, pol_rows_over 0, wdl_rows_over 0. The tail arm is the only
    thing that can produce this verdict, and the assertions below say so.
    """
    rng = np.random.default_rng(11)
    n = 2000
    ref = rng.normal(0, 4.0, size=(n, 128)).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(n, 3)).astype(np.float32)
    ctl = _ulp_scale_control(ref, rng)
    ctl_w = _ulp_scale_control(ref_w, rng)

    aot = _ulp_scale_control(ref, rng)
    hurt = int(n * 0.08)
    aot[:hurt] = ref[:hurt] + rng.normal(0, 0.16, size=(hurt, 128)).astype(np.float32)

    ok, detail, _, _ = _gate(aot, ctl_w.copy(), ref, ref_w, ctl, ctl_w)
    arms = _arms(detail)
    not_tail_only = (
        f"scenario is not tail-only: a mean arm already fires, so this test "
        f"would pass with the tail arm deleted: {detail}"
    )
    assert arms["pol_mean"] < 2.0, not_tail_only
    assert arms["wdl_mean"] < 2.0, not_tail_only
    not_the_tail_arm = (
        f"the row-exceedance arm fired, so the tail arm is not what caught "
        f"this and deleting it would leave the test green: {detail}"
    )
    assert arms["pol_rows_over"] == 0, not_the_tail_arm
    assert arms["wdl_rows_over"] == 0, not_the_tail_arm
    assert "FLOOR-DIVERGENCE" not in detail, detail
    assert arms["pol_tail"] > 2.0, detail
    assert not ok, f"tail-only row damage passed the gate: {detail}"


def test_cli_tv_ratio_max_reaches_verify_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⚑ The CLI->verify_packages leg, which no end-to-end test can see.

    Hard-coding a constant at this call site left every other test green.
    """
    seen: dict[str, object] = {}

    def _recorder(**kw: object) -> tuple[int, int, list[object]]:
        seen.update(kw)
        return 1, 0, []

    monkeypatch.setattr(MOD, "verify_packages", _recorder)
    monkeypatch.setattr(MOD, "build_reference_model", lambda *a, **k: _StubModel())
    monkeypatch.setattr(MOD, "load_model_config", lambda p: _StubModel())
    monkeypatch.setattr(MOD, "load_checkpoint_state_dict", lambda p: {})
    monkeypatch.setattr(MOD, "_require_cuda", lambda action: None)
    MOD.main([
        "--checkpoint", "x.pt", "--verify-only", "--out-dir", str(tmp_path),
        "--buckets", "8", "--max-batch", "8",
        "--tv-ratio-max", "3.75", "--argmax-min", "0.42", "--verify-n", "7",
    ])
    assert seen.get("tv_ratio_max") == pytest.approx(3.75), seen
    assert seen.get("argmax_min") == pytest.approx(0.42), seen
    assert seen.get("verify_n") == 7, seen


def test_end_to_end_the_control_actually_reaches_the_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⚑ Computing the control and then discarding it at the call site is
    invisible to every verdict-based test — the ULP floor covers for it. The
    floor's PROVENANCE is what distinguishes them, so assert on that.

    ⚑⚑ AND ON CPU THE PROVENANCE IS `ulp-only(ctl-degenerate)`, NOT
    `shape+ulp` — which is F8 made observable rather than a weaker assertion.
    `eager_batch_shape_control` is BITWISE identical to the full batch on CPU,
    so the degeneracy escape fires and the cross-check never runs. The label
    still discriminates: `ulp-only(ctl-degenerate)` can ONLY be produced when a
    control was passed in and measured, while `ctl is None` reads a bare
    `ulp-only`. Asserting the bare string would have gone green on a discarded
    control; asserting this one cannot.

    It also records the standing gap: **every CPU verify runs without the
    cross-check**, so the floor's only self-check on this machine is the
    absolute plausibility bound.
    """
    rows = _run_verify(tmp_path, monkeypatch, offset=0.0)[2]
    assert (
        "floor=pol:ulp-only(ctl-degenerate) wdl:ulp-only(ctl-degenerate)"
        in rows[0][2]
    ), (
        f"the batch-shape control never reached _compare_bucket: {rows}"
    )


def test_the_ulp_floor_magnitude_quoted_in_the_docstring_is_reproducible() -> None:
    """⚑ BANK THE DUMP. RE-MEASURED after the raw-ULP fix; the old 0.0202 is gone.

    This is the one calibration number in the gate that CI hardware can
    reproduce. The CUDA readings (0.0176 control / 0.01737 AOT at bucket 1190)
    cannot be, they were taken against the CENTRED floor, and they are labelled
    as provenance rather than calibration in the source.

    ⚑ AND THIS ARRAY IS NOT THE PRODUCTION REGIME — say so, because a number
    banked here has been read as a production floor before. These are zero-mean
    sd-6 synthetic logits; the real net's live policy logits have mean -6.08 and
    within-row spread 3.03, and the honest floors measured on it (bt4heads
    iter100, CPU bf16, batches of 64) are:

        head    floor mean            floor tail
        policy  5.35e-3 .. 5.99e-3    1.02e-2 .. 1.28e-2
        wdl     8.08e-3 .. 1.27e-2    3.42e-2 .. 5.55e-2

    which is 6x BELOW what this synthetic array reads. The gate is scale-free,
    so that is not a problem — but quoting this number as "the production floor"
    would be, and an earlier revision of this file did exactly that.
    """
    rng = np.random.default_rng(0)
    ref = torch.from_numpy(
        (rng.normal(0, 1.0, size=(256, 1858)) * 6.0).astype(np.float32)
    ).to(torch.bfloat16).float().numpy()
    p = _probs(ref)
    top1 = float(p.max(axis=-1).mean())
    floor = mean_row_tv(_probs(bf16_ulp_perturbation(ref, seed=0)), p)

    assert 0.50 < top1 < 0.65, f"sharpness regime moved: top-1 {top1:.4f}"
    assert 0.025 < floor < 0.040, (
        f"ULP floor {floor:.4f} is outside the measured 0.0323 on this array; "
        "the tv_ratio_max default was calibrated against this magnitude"
    )
    assert floor < MOD._FLOOR_IMPLAUSIBLE_TV, (
        f"the plausibility bound {MOD._FLOOR_IMPLAUSIBLE_TV} is below a floor "
        f"a HEALTHY sharp net produces ({floor:.4f}) — it would fail every "
        "bucket"
    )


def test_the_ulp_floor_is_shift_COVARIANT_because_bf16_ERROR_IS() -> None:
    """⚑⚑ RE-DERIVED. This test used to pin shift-INVARIANCE; that was wrong.

    What it pins now: moving the whole logit row must move the floor, because
    bf16 rounds the RAW value. A net whose head bias actually drifted emits
    larger-magnitude logits, and those really do round more coarsely — the error
    is a property of the number, not of the softmax it feeds.

    The old argument for invariance — "adding a constant leaves the softmax and
    the measured AOT-vs-eager TV unchanged, so the floor must not move" —
    conflates two different operations. Adding a constant to an ALREADY-COMPUTED
    logit array is a post-hoc edit no bf16 pipeline performs; the AOT and eager
    sides would both round the shifted value and their difference would grow
    with it. So the correct property is COVARIANCE.

    Measured here on bf16-rounded sd-6 logits: the floor is flat inside a binade
    and steps up by ~2x across each one, so a +100 offset (which drags every
    logit into the 64..128 binade) raises it far above the unshifted value. The
    assertion is one-sided and generous — this is a claim about the DIRECTION of
    the dependence, not a calibration.
    """
    rng = np.random.default_rng(0)
    base = torch.from_numpy(
        (rng.normal(0, 1.0, size=(256, 512)) * 6.0).astype(np.float32)
    ).to(torch.bfloat16).float().numpy()

    floors = [
        mean_row_tv(_probs(bf16_ulp_perturbation(base + off, seed=0)),
                   _probs(base + off))
        for off in (0.0, 100.0)
    ]
    assert floors[1] > 3 * floors[0], (
        f"the floor did NOT grow with the logits' magnitude: {floors}. bf16 "
        "error scales with the raw value; a floor that ignores the offset is "
        "the centred-relative form this test exists to keep out."
    )
    # ⚑ And the SOFTMAX really is unchanged by the shift, so the covariance is a
    # statement about the floor's physics and not an artifact of the fixture
    # having moved the distribution.
    assert np.allclose(_probs(base), _probs(base + 100.0), atol=1e-6), (
        "premise: a common offset leaves the softmax alone"
    )


def test_single_row_corruption_fails_even_though_p99_misses_it() -> None:
    """⚑ p99 ignores damage in <1% of rows — a boundary-row kernel bug.

    `pol_rows_over` is what carries this: a COUNT of rows whose TV exceeds
    `_ROW_EXCEED_K` times the floor's own p99, gated at zero. It is not an
    extreme-value statistic that grows with N the way the original whole-array
    max did — the threshold is in units of the floor, and the count of rows
    above it is zero on a healthy package at every bucket size (re-derived in
    `test_every_surviving_arm_reads_about_one_on_a_healthy_production_package`).
    """
    rng = np.random.default_rng(21)
    n = 512
    ref = rng.normal(0, 4.0, size=(n, 256)).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(n, 3)).astype(np.float32)
    aot = ref + rng.normal(0, 0.01, size=ref.shape).astype(np.float32)
    aot[-1] = rng.normal(0, 4.0, size=256)  # ONE boundary row, totally wrong

    # Both other arms are blind to it, which is exactly why this arm exists.
    assert mean_row_tv(_probs(aot), _probs(ref)) < 0.01
    assert float(np.quantile(row_tv(_probs(aot), _probs(ref)), 0.99)) < 0.10

    ok, detail, matches, rows = _gate(aot, ref_w.copy(), ref, ref_w)
    assert matches / rows > 0.99, (
        f"argmax is also blind here by design ({matches}/{rows}) — it sits far "
        "above the 0.90 floor, which is why a TV arm has to carry this"
    )
    assert _arms(detail)["pol_rows_over"] == 1, (
        f"exactly the one damaged row should be flagged: {detail}"
    )
    assert not ok, f"single-row corruption passed the gate: {detail}"


def test_an_infinite_ratio_threshold_is_REJECTED(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """argparse's float() happily returns inf, and `inf <= 0.0` is False."""
    monkeypatch.setattr(MOD, "build_reference_model", lambda *a, **k: _StubModel())
    monkeypatch.setattr(MOD, "load_model_config", lambda p: _StubModel())
    monkeypatch.setattr(MOD, "load_checkpoint_state_dict", lambda p: {})
    monkeypatch.setattr(MOD, "_require_cuda", lambda action: None)
    for bad in ("inf", "nan"):
        rc = MOD.main([
            "--checkpoint", "x.pt", "--verify-only", "--out-dir", str(tmp_path),
            "--buckets", "8", "--max-batch", "8", "--tv-ratio-max", bad,
        ])
        assert rc == 2, f"--tv-ratio-max {bad} was accepted (rc={rc})"


def test_a_healthy_package_passes_EVERY_arm_at_a_large_bucket() -> None:
    """⚑ The false-positive guard, and the reason each floor uses a MATCHED
    functional.

    A max over 512 rows is naturally 3-5x the mean of the same rows. So an arm
    that compared a per-row extreme against the floor's MEAN would read ~4x on
    this perfectly healthy package and FAIL — a gate that rejects good packages
    at big buckets, which is exactly the shape of the original defect. Every
    ratio arm's denominator is the same statistic as its numerator, and the
    row-exceedance arm's threshold is the floor's TAIL rather than its mean for
    the same reason.
    """
    rng = np.random.default_rng(31)
    n = 512
    ref = (rng.normal(0, 1.0, size=(n, 1024)) * 5.0).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(n, 3)).astype(np.float32)
    noise = np.float32(2.0) ** -8  # exactly the floor's own scale
    aot = ref * (1.0 + rng.normal(0, noise, size=ref.shape).astype(np.float32))
    aot_w = ref_w * (1.0 + rng.normal(0, noise, size=ref_w.shape).astype(np.float32))

    rows = row_tv(_probs(aot), _probs(ref))
    assert float(np.max(rows)) > 2.5 * float(np.mean(rows)), (
        "scenario has no max/mean spread, so it cannot detect a mismatched "
        f"denominator (max {np.max(rows):.4g} vs mean {np.mean(rows):.4g})"
    )
    ok, detail, _, _ = _gate(aot, aot_w, ref, ref_w)
    assert ok, f"a healthy package failed at a large bucket: {detail}"


def test_every_surviving_arm_reads_about_one_on_a_healthy_production_package() -> None:
    """⚑⚑ BANK THE DUMP. The healthy null is a MEASUREMENT; this re-derives it.

    There is no `_ARM_NULL` table any more, and its deletion is the claim under
    test: the three surviving arms are supposed to read ~1.0 *unnormalized*, so
    that one `--tv-ratio-max` means the same thing on all of them. The two arms
    that could not (wdl_tail, wdl_max — a p99 and a max over THREE columns,
    healthy spread 2.4x their own mean) were removed rather than propped up by a
    per-arm divisor, because a divisor big enough to stop their false failures
    is a divisor big enough to stop their true ones.

    Re-derived here on PRODUCTION-SHAPED arrays — policy 4672 wide with 1858
    live and 2814 sentinel slots, WDL 3 wide — which is the shape the deleted
    table was never measured on and the shape the sentinel-mass defect lived in.

    ⚑⚑ THE PREVIOUS VERSION OF THIS TEST WAS CIRCULAR AND COULD NOT HAVE CAUGHT
    THE F1 DEFECT. It built the healthy numerator as
    ``bf16_ulp_perturbation(..., seed=1000+seed)`` — the DENOMINATOR'S OWN
    FUNCTION at another seed. Every arm then reads ~1.0 no matter what that
    function computes, including when it computes something with no physical
    meaning: it measured self-consistency, not agreement with a real package
    discrepancy. Compounding it, every generator drew ``rng.normal(0, s)``, i.e.
    ZERO-MEAN logits — the single regime in which the centred and raw
    perturbations coincide, so the defect was invisible by construction.

    Two things are fixed here. The healthy numerator is now written OUT in this
    file (`_independent_abs_ulp` / `_independent_rel_jitter`), sharing no code
    with the gate; and the sweep includes the LARGE-COMMON-OFFSET, SMALL-SPREAD
    regime the real WDL head lives in (row mean -8.6, within-row spread 1.0),
    where the two perturbation forms differ by 12-22x.

    RE-MEASURED 2026-08-15 on the fixed source, buckets {8, 64, 512} x 8 seeds,
    n=24 per arm per cell:

        regime      model       pol_mean            pol_tail            wdl_mean
                                mean/med/worst      mean/med/worst      mean/med/worst
        zero-mean   abs-ULP     0.990/0.995/1.100   1.000/0.990/1.290   1.017/1.010/1.390
        zero-mean   rel-2^-8    0.615/0.605/0.750   0.735/0.665/1.110   0.641/0.625/1.010
        production  abs-ULP     0.995/1.000/1.090   0.994/0.980/1.220   1.057/1.010/2.030
        production  rel-2^-8    0.590/0.580/0.690   0.848/0.875/1.090   0.650/0.605/1.840

    and `pol_rows_over` / `wdl_rows_over` are 0 in every cell.

    ⚑ Read the abs-ULP rows, not the rel-2^-8 ones, for calibration: a raw ULP
    is a ±1-in-the-last-place move while ``z*(1+N(0,2^-8))`` is a gaussian of
    that scale, ~0.6x as large in TV, so it flatters the gate. The honest
    headline is the abs-ULP worst, **2.03 at bucket 8 on the WDL arm** — see
    `--tv-ratio-max`'s help for the per-bucket margin and the residual
    false-fail rate that implies.

    On the OLD source the production/abs-ULP cell read wdl_mean mean 12.6,
    worst 26.4 — a package that is one bf16 ULP off, the best a package can
    physically be, failing 24/24 trials.
    """
    seen: dict[tuple[str, str], dict[str, list[float]]] = {}
    for regime, pol_off, pol_sd, wdl_off, wdl_sd in (
        ("zero-mean", 0.0, 5.0, 0.0, 1.0),
        ("production", -6.0, 3.0, -8.6, 1.0),
    ):
        for model, jitter in (
            ("abs-ULP", _independent_abs_ulp), ("rel-2^-8", _independent_rel_jitter),
        ):
            cell = seen.setdefault((regime, model), {})
            for n in (8, 64, 512):
                for seed in range(8):
                    rng = np.random.default_rng(seed)
                    live = (rng.normal(0, 1.0, size=(n, LIVE_LOGITS)) * pol_sd
                            + pol_off).astype(np.float32)
                    ref_w = (rng.normal(0, 1.0, size=(n, 3)) * wdl_sd
                             + wdl_off).astype(np.float32)
                    jr = np.random.default_rng(9000 + seed)
                    _, detail, _, _ = _compare_bucket(
                        aot_pol=_sentinel_padded(jitter(live, jr)),
                        aot_wdl=jitter(ref_w, jr),
                        ref_pol=_sentinel_padded(live), ref_wdl=ref_w,
                        ctl_pol=None, ctl_wdl=None, tv_ratio_max=1e12,
                        floor_seed=seed,
                    )
                    for arm, val in _arms(detail).items():
                        cell.setdefault(arm, []).append(val)

    for key, cell in seen.items():
        assert set(cell) == {
            "pol_mean", "pol_tail", "wdl_mean", "pol_rows_over", "wdl_rows_over",
        }, f"arm set drifted at {key}: {sorted(cell)}"

    # ⚑ The abs-ULP cells are the physical model, so they carry the "reads ~1.0
    # unnormalized" claim — the claim that deleting _ARM_NULL made. The
    # rel-2^-8 cells only have to stay UNDER it; a gaussian at the same scale is
    # a smaller move, and asserting ~1.0 on them would be asserting a
    # coincidence.
    for arm in ("pol_mean", "pol_tail", "wdl_mean"):
        for regime in ("zero-mean", "production"):
            vals = seen[(regime, "abs-ULP")][arm]
            mean = float(np.mean(vals))
            assert 0.85 < mean < 1.20, (
                f"[{regime}/abs-ULP] arm {arm} reads {mean:.3f} on a HEALTHY "
                "production-shaped package, not ~1.0 — one --tv-ratio-max no "
                "longer means the same thing on every arm, which is what "
                "deleting _ARM_NULL asserted"
            )
            assert max(vals) < 2.5, (
                f"[{regime}/abs-ULP] arm {arm} peaked at {max(vals):.2f} on a "
                "HEALTHY package against a 2.0 default bound"
            )
            rel = seen[(regime, "rel-2^-8")][arm]
            assert float(np.mean(rel)) < 1.20, (
                f"[{regime}/rel-2^-8] arm {arm} mean {np.mean(rel):.3f}"
            )

    # ⚑⚑ THE F1 REGRESSION ASSERTION, and the reason the production regime is
    # here at all. On the centred source the WDL arm read a mean of 12.6 in this
    # cell. Anything near that means the floor is head-dependent again.
    wdl_prod = seen[("production", "abs-ULP")]["wdl_mean"]
    assert float(np.mean(wdl_prod)) < 1.5, (
        f"wdl_mean reads {np.mean(wdl_prod):.2f} on a package that is EXACTLY "
        "one bf16 ULP off, at the real WDL head's offset/spread ratio. That is "
        "the F1 defect: a floor derived from centred logits is 12-22x too small "
        "on this head, so the best a package can physically be reads as a fail."
    )

    # The count arms are gated at ZERO, so their null is not "about one", it is
    # "never". A single false exceedance here would make the gate reject good
    # packages outright, with no threshold left to absorb it.
    for key, cell in seen.items():
        for arm in ("pol_rows_over", "wdl_rows_over"):
            assert max(cell[arm]) == 0, (
                f"{arm} fired on a healthy package at {key}: {cell[arm]} — "
                f"_ROW_EXCEED_K = {MOD._ROW_EXCEED_K} is too tight"
            )


# --------------------------------------------------------------------------
# ⚑⚑ THE PRODUCTION WIDTH — where the gate was arithmetically unable to fail
#
# Everything above this line runs on narrow synthetic arrays. The blocking
# defect lived ONLY at the real policy width: `_policy_output_full` emits 4672
# slots of which 2814 are a -1e9 sentinel, `bf16_ulp_perturbation` mean-centred
# the whole row, and that mean sat near -6.0e8. Every real logit was then
# centred to ~+6.0e8, a 2^-8 relative nudge moved it by ~2.3e6, and the floor
# saturated at 0.9719 against 0.0189 on the native 1858-wide array — 51x. A row
# TV is bounded by 1, so no policy arm could exceed even a 1.5 bound: a
# completely wrong policy verified clean at pol_mean x1.17 with argmax 0/256.
# --------------------------------------------------------------------------

def test_the_ulp_floor_is_not_inflated_by_sentinel_padding() -> None:
    """⚑⚑ THE regression test for the blocking defect. Pre-fix this read 51x.

    Softmax cannot see the sentinel slots at all, so widening a logit row with
    them must not change the floor by any amount. Measured on the fixed source:
    0.01954 native vs 0.01753 padded at 64 rows, 0.9944 agreement at 256.

    ⚑ ROW COUNT MATTERS AND 64 IS NOT ENOUGH. The padded and native calls draw
    DIFFERENT sign patterns (`rng.choice(size=z.shape)` is shape-dependent), so
    the two floors are independent estimates of the same quantity, and a row TV
    at this sharpness is dominated by the sign landed on the row's top few
    logits. At 64 rows the padded/native ratio spans [0.897, 1.031] over six
    seeds — the estimator's own noise breaches a 10% band on its own. At 256 it
    spans [0.965, 0.997] and at 1024 [0.978, 1.010]. So this runs at 256 rows,
    where a 10% band is a real assertion about padding rather than about
    sampling. The defect it guards is 51x; no tolerance in this range is load
    bearing against that.
    """
    rng = np.random.default_rng(0)
    native = (rng.normal(0, 1.0, size=(256, LIVE_LOGITS)) * 5.0).astype(np.float32)
    padded = _sentinel_padded(native)

    f_native = mean_row_tv(_probs(bf16_ulp_perturbation(native, seed=0)), _probs(native))
    f_padded = mean_row_tv(_probs(bf16_ulp_perturbation(padded, seed=0)), _probs(padded))

    assert f_native > 0.0, "premise: the native floor is non-degenerate"
    assert f_padded == pytest.approx(f_native, rel=0.10), (
        f"the ULP floor moved with sentinel padding: native {f_native:.5f} vs "
        f"padded {f_padded:.5f} (x{f_padded / f_native:.2f}). A floor that "
        "grows with the mask width caps every policy arm below its own bound."
    )


def test_a_wrong_policy_FAILS_at_the_production_width() -> None:
    """⚑⚑ THE false-negative test. On the pre-fix source this PASSED.

    An INDEPENDENT random draw in the live columns is as wrong as a policy can
    be — it shares nothing with the reference but the mask. WDL is held healthy
    at exactly bf16-ULP scale, so the verdict has to be produced by the policy
    arms alone; if it were the WDL arms firing, this test would say nothing
    about the padding at all.

    Measured on the fixed source: pol_mean x52.70, pol_tail x30.82,
    pol_rows_over 256/256, wdl_mean x1.10 — and argmax 0/256, which is the
    independent confirmation that the two arrays really are unrelated.
    """
    rng = np.random.default_rng(17)
    live = (rng.normal(0, 1.0, size=(256, LIVE_LOGITS)) * 5.0).astype(np.float32)
    ref_pol = _sentinel_padded(live)
    aot_pol = _sentinel_padded(
        (rng.normal(0, 1.0, size=(256, LIVE_LOGITS)) * 5.0).astype(np.float32)
    )
    ref_wdl = rng.normal(0, 1.0, size=(256, 3)).astype(np.float32)
    ulp = np.float32(2.0) ** -8
    aot_wdl = (ref_wdl * (1.0 + rng.normal(0, ulp, size=ref_wdl.shape))).astype(np.float32)

    ok, detail, matches, rows = _gate(aot_pol, aot_wdl, ref_pol, ref_wdl)
    arms = _arms(detail)

    assert matches <= rows // 100, (
        f"premise failed — the two policies agree on {matches}/{rows} argmaxes, "
        "so they are not independent draws and this is not the wrong-policy case"
    )
    assert arms["wdl_mean"] < 2.0, (
        f"WDL was supposed to be healthy; if it fires, the policy arms are not "
        f"what produced the verdict: {detail}"
    )
    assert not ok, (
        f"a COMPLETELY WRONG POLICY verified clean at the production width: {detail}"
    )


def test_the_floor_is_shift_COVARIANT_at_the_production_width() -> None:
    """⚑ RE-DERIVED, same reversal as the narrow case, and it pins the OTHER
    half of the F1 fix: the mask must not participate in the covariance.

    The floor must grow when the LIVE logits move, because bf16 rounds the raw
    value. It must NOT move at all when only the mask width changes — the
    sentinels are excluded by the `live` mask rather than by a centre, so
    padding invariance now holds by CONSTRUCTION rather than by arithmetic
    luck. Those two are separate claims and this test makes both.
    """
    rng = np.random.default_rng(2)
    live = (rng.normal(0, 1.0, size=(128, LIVE_LOGITS)) * 5.0).astype(np.float32)
    padded = _sentinel_padded(live)
    shifted = padded.copy()
    shifted[:, :LIVE_LOGITS] += 500.0  # sentinels deliberately left where they are

    base = mean_row_tv(_probs(bf16_ulp_perturbation(padded, seed=0)), _probs(padded))
    moved = mean_row_tv(_probs(bf16_ulp_perturbation(shifted, seed=0)), _probs(shifted))

    assert base > 0.0, "premise: the floor is non-degenerate"
    assert moved > 10.0 * base, (
        f"the floor did not follow a +500 shift of the LIVE logits: {base:.6f} "
        f"-> {moved:.6f}. Every live logit is now in the 256..512 binade, whose "
        "bf16 spacing is ~64x the unshifted row's; the floor has to say so."
    )

    # ⚑ Mask width, by contrast, must change NOTHING — the sentinels take no
    # part. Same live block, same seed, so the sign draw is the only difference
    # and it is shape-dependent; the tolerance is for that, not for the mask.
    narrow = mean_row_tv(_probs(bf16_ulp_perturbation(live, seed=0)), _probs(live))
    assert narrow == pytest.approx(base, rel=0.10), (
        f"the floor moved with the MASK width: native {narrow:.6f} vs padded "
        f"{base:.6f}. Softmax cannot see the sentinels, so the floor must not."
    )


def test_sentinel_slots_stay_dead_after_perturbation() -> None:
    """⚑ The masked slots must carry exactly zero probability, still.

    RE-DERIVED for the raw-ULP fix. There is no centring any more, so the
    sentinels are not shifted either: the contract is now the strictest one
    available — they come back **bit-identical**. If a perturbation ever reached
    them, mass would leak into illegal moves and the floor would be measuring
    the mask rather than the policy.

    ⚑ AND THE CONTRACT HAS TO BE ASSERTED ON THE LOGITS, NOT ONLY ON THE
    PROBABILITIES. Measured 2026-08-15: perturbing the sentinels too leaves the
    softmax BIT-IDENTICAL and the sentinel mass at exactly 0, because one ULP of
    -1e9 is ~3.9e6 and `exp` of either is 0. So a probability-space assertion
    alone cannot distinguish the two — that mutant SURVIVED the first round of
    this test. It is harmless only because the sentinel is nine orders of
    magnitude below the live range; the moment that margin shrinks it stops
    being harmless, which is exactly what a contract assertion is for.
    """
    rng = np.random.default_rng(6)
    live = (rng.normal(0, 1.0, size=(64, LIVE_LOGITS)) * 5.0).astype(np.float32)
    padded = _sentinel_padded(live)
    perturbed = bf16_ulp_perturbation(padded, seed=0)
    p = _probs(perturbed)

    dead = p[:, LIVE_LOGITS:]
    assert dead.shape[1] == FULL_POLICY - LIVE_LOGITS == 2814
    assert float(np.max(np.abs(dead))) == 0.0, (
        f"probability mass leaked into the {dead.shape[1]} sentinel slots: "
        f"max {float(np.max(np.abs(dead))):.3e}"
    )
    assert p.sum(axis=-1) == pytest.approx(np.ones(64), rel=0, abs=1e-6), (
        "rows no longer normalize to 1 after the perturbation"
    )

    # Premise for the contract check: every live entry is inside the 80-logit
    # window, so `live` selects exactly the 1858 real columns and the centre is
    # the live mean.
    assert float(np.max(live.max(axis=-1) - live.min(axis=-1))) < 80.0
    assert np.array_equal(perturbed[:, LIVE_LOGITS:], padded[:, LIVE_LOGITS:]), (
        "the sentinel slots were PERTURBED; one bf16 ULP of -1e9 is ~3.9e6, "
        "which the softmax cannot see today and will not always be able to "
        "ignore. With no centring they must come back BIT-IDENTICAL."
    )


def test_a_single_corrupted_row_fails_even_though_every_mean_arm_reads_below_one() -> None:
    """⚑⚑ THE test that makes `_ROW_EXCEED_K` non-vacuous.

    One wrong row in 2048 is the boundary-row kernel bug the count arm exists
    for, and the dilution does not merely hide it — it points the WRONG WAY.
    Measured: pol_mean x0.04, pol_tail x0.00, wdl_mean x0.00, i.e. every ratio
    arm reads BELOW its healthy null, because the other 2047 rows are bit-exact
    and drag the numerator under the floor. No `tv_ratio_max` above 0 can fail
    this; only a count of rows over the floor can. `pol_rows_over` reads exactly
    1 — the damaged row and nothing else.

    Argmax is blind here too (2047/2048 = 0.9995, far above any sane bound), so
    there is no other criterion covering for the count arm.
    """
    aot, aot_w, ref, ref_w = _one_bad_row_case()
    ok, detail, matches, rows = _gate(aot, aot_w, ref, ref_w)
    arms = _arms(detail)

    for arm in ("pol_mean", "pol_tail", "wdl_mean"):
        assert arms[arm] < 1.0, (
            f"the ratio arm {arm} reads x{arms[arm]:.2f} >= 1.0, so it could "
            f"have produced this verdict and the count arm is untested: {detail}"
        )
    assert matches / rows > 0.99, f"argmax is meant to be blind here: {detail}"
    assert arms["pol_rows_over"] == 1, (
        f"expected exactly the one damaged row to be flagged: {detail}"
    )
    assert not ok, f"a single corrupted row passed the gate: {detail}"


def test_the_row_exceedance_arm_is_not_loosened_by_tv_ratio_max() -> None:
    """⚑ The count arm is in units of the package's own floor, so the ratio
    knob must not reach it. Otherwise `--tv-ratio-max 1e6` — an entirely
    plausible way to silence a noisy arm — would silently retire the only
    criterion with power against single-row damage.
    """
    aot, aot_w, ref, ref_w = _one_bad_row_case()
    ok, detail, _, _ = _gate(aot, aot_w, ref, ref_w, tv_ratio_max=1e6)
    assert _arms(detail)["pol_rows_over"] == 1, detail
    assert not ok, (
        f"tv_ratio_max=1e6 disabled the row-exceedance arm: {detail}"
    )


def _one_bad_row_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2048 bit-identical rows and ONE independent draw. Shared by two tests."""
    rng = np.random.default_rng(23)
    n = 2048
    ref = (rng.normal(0, 1.0, size=(n, 256)) * 4.0).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(n, 3)).astype(np.float32)
    aot = ref.copy()
    aot[1234] = (rng.normal(0, 1.0, size=256) * 4.0).astype(np.float32)
    return aot, ref_w.copy(), ref, ref_w


def test_the_floor_divergence_guard_fires_and_fails_the_bucket() -> None:
    """⚑⚑ The guard that WOULD have caught the sentinel-mass defect (it showed
    as ~45x). The two floor estimates are independent derivations of the same
    quantity; on real hardware they agree to 15%. A 5x gap means one of them is
    broken, and a broken floor is the one failure this gate cannot self-detect —
    it rescales every arm, in the PASSING direction when the floor is too big.

    So it must FAIL, not warn. Reporting a divergence and passing anyway is the
    "accepted then silently ignored" defect the gate exists to catch.

    Paired negative control below: a control that agrees with the ULP floor must
    NOT set the note and must NOT fail. Without it, `note = "..."` unconditional
    would pass this test.
    """
    rng = np.random.default_rng(29)
    ref = (rng.normal(0, 1.0, size=(128, 256)) * 4.0).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(128, 3)).astype(np.float32)
    ulp = np.float32(2.0) ** -8
    aot = (ref * (1.0 + rng.normal(0, ulp, size=ref.shape))).astype(np.float32)
    aot_w = (ref_w * (1.0 + rng.normal(0, ulp, size=ref_w.shape))).astype(np.float32)
    healthy_ctl_w = (ref_w * (1.0 + rng.normal(0, ulp, size=ref_w.shape))).astype(np.float32)

    # A "control" that shares nothing with the reference: floor x86.4.
    broken_ctl = (rng.normal(0, 1.0, size=ref.shape) * 4.0).astype(np.float32)
    ok, detail, _, _ = _gate(aot, aot_w, ref, ref_w, broken_ctl, healthy_ctl_w)
    assert "FLOOR-DIVERGENCE" in detail, (
        f"a floor whose two estimates disagree 86x was not reported: {detail}"
    )
    assert not ok, f"a divergent floor was reported and then ignored: {detail}"
    arms = _arms(detail)
    assert all(arms[a] <= 2.0 for a in ("pol_mean", "pol_tail", "wdl_mean")), (
        f"every ratio arm must be PASSING here, or the verdict could have come "
        f"from an arm rather than from the guard: {detail}"
    )
    assert arms["pol_rows_over"] == 0, detail
    assert arms["wdl_rows_over"] == 0, detail

    # ⚑ PAIRED NEGATIVE CONTROL. Same package, a control that agrees.
    healthy_ctl = (ref * (1.0 + rng.normal(0, ulp, size=ref.shape))).astype(np.float32)
    ok_ctl, detail_ctl, _, _ = _gate(aot, aot_w, ref, ref_w, healthy_ctl, healthy_ctl_w)
    assert "FLOOR-DIVERGENCE" not in detail_ctl, (
        f"the guard fires on an AGREEING control, so it cannot pass: {detail_ctl}"
    )
    assert ok_ctl, detail_ctl


def test_the_floor_cross_check_covers_the_TAIL_not_only_the_mean() -> None:
    """⚑ F7. `hi, lo = max/min(u_mean, c_mean)` cross-checked the MEANS only,
    while the returned tail was an unchecked `max(u_tail, c_tail)` — and the
    TAIL is what sets `_ROW_EXCEED_K`'s threshold. So a floor bug that inflated
    only the tail disabled BOTH exceedance arms, the gate's only power against
    single-row damage, with nothing in the detail line saying so.

    The scenario is built to be tail-ONLY, or it would say nothing about which
    half of the cross-check fired: measured, the two mean estimates agree to
    x1.00 while the tails differ x13.0. It also has to stay UNDER
    `_FLOOR_IMPLAUSIBLE_TV` (measured tail 0.166 against a 0.25 bound), because
    the absolute bound would otherwise catch it and the tail cross-check could
    be deleted with this test still green.
    """
    rng = np.random.default_rng(13)
    n = 500
    ref = (rng.normal(0, 1.0, size=(n, 128)) * 2.0).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(n, 3)).astype(np.float32)
    ctl = _ulp_scale_control(ref, rng)
    ctl[:int(n * 0.02)] += rng.normal(0, 0.5, size=(int(n * 0.02), 128))
    ctl = ctl.astype(np.float32)
    ctl_w = _ulp_scale_control(ref_w, rng)
    aot = _ulp_scale_control(ref, rng)
    aot_w = _ulp_scale_control(ref_w, rng)

    ok, detail, _, _ = _gate(aot, aot_w, ref, ref_w, ctl, ctl_w)
    arms = _arms(detail)
    assert "FLOOR-DIVERGENCE[tail]" in detail, (
        f"the TAIL estimates differ 13x and nothing reported it: {detail}"
    )
    assert "FLOOR-DIVERGENCE[mean]" not in detail, (
        f"scenario is not tail-only — the MEAN cross-check fired too, so this "
        f"test would stay green with the tail half deleted: {detail}"
    )
    assert "FLOOR-IMPLAUSIBLE" not in detail, (
        f"the absolute plausibility bound caught it, so the tail cross-check is "
        f"again not what produced the verdict: {detail}"
    )
    for arm in ("pol_mean", "pol_tail", "wdl_mean"):
        assert arms[arm] <= 2.0, f"a ratio arm produced the verdict: {detail}"
    assert arms["pol_rows_over"] == 0, detail
    assert arms["wdl_rows_over"] == 0, detail
    assert not ok, f"a floor whose TAIL is 13x divergent was passed: {detail}"

    # ⚑ PAIRED NEGATIVE CONTROL: the same package, a control without the tail.
    clean_ctl = _ulp_scale_control(ref, rng)
    ok_clean, detail_clean, _, _ = _gate(aot, aot_w, ref, ref_w, clean_ctl, ctl_w)
    assert "FLOOR-DIVERGENCE" not in detail_clean, detail_clean
    assert ok_clean, detail_clean


def test_the_degeneracy_escape_is_ONE_SIDED() -> None:
    """⚑⚑ F3. The escape must fire only when the SHAPE CONTROL is the small one.

    When it tested `min(ulp, ctl)` instead, a SMALL ULP ESTIMATE retired the
    cross-check — and the code then took `max(ulp, ctl)`, so the inflated
    control became the floor unchallenged. Demonstrated at production shapes: a
    tiny-magnitude WDL reference (raw ULP floor 1.7e-6, well under
    `_FLOOR_DEGENERATE_TV`) plus a control 8800x larger made a COMPLETELY WRONG
    AOT WDL read `ok=True wdl_mean=x1.00 wdl_rows_over=0`.

    One-sided, the same inputs report a divergent floor on both statistics and
    FAIL. The policy side is held healthy so the verdict cannot come from it.
    """
    rng = np.random.default_rng(41)
    n = 64
    live = (rng.normal(0, 1.0, size=(n, LIVE_LOGITS)) * 3.0 - 6.0).astype(np.float32)
    ref_pol = _sentinel_padded(live)
    ref_w = rng.normal(0, 1e-3, size=(n, 3)).astype(np.float32)

    ulp_floor = mean_row_tv(_probs(bf16_ulp_perturbation(ref_w, seed=0)), _probs(ref_w))
    assert ulp_floor < MOD._FLOOR_DEGENERATE_TV, (
        f"premise: the ULP estimate ({ulp_floor:.3e}) must be the SMALL one, or "
        "this scenario cannot distinguish a one-sided escape from a two-sided one"
    )

    ctl_w = (ref_w + rng.normal(0, 0.05, size=ref_w.shape)).astype(np.float32)
    aot_w = rng.normal(0, 1.0, size=(n, 3)).astype(np.float32)  # COMPLETELY wrong
    aot_pol = _sentinel_padded(_ulp_scale_control(live, rng))
    ctl_pol = _sentinel_padded(_ulp_scale_control(live, rng))

    ok, detail, _, _ = _gate(aot_pol, aot_w, ref_pol, ref_w, ctl_pol, ctl_w)
    arms = _arms(detail)
    for arm in ("pol_mean", "pol_tail"):
        assert arms[arm] <= 2.0, (
            f"the policy side was meant to be healthy: {detail}"
        )
    assert "FLOOR-DIVERGENCE" in detail, (
        f"a control 8800x the ULP estimate was accepted silently — the escape "
        f"is still two-sided: {detail}"
    )
    assert not ok, f"a completely wrong AOT WDL verified clean: {detail}"


def test_a_degenerate_control_is_NOT_allowed_to_widen_the_floor() -> None:
    """⚑ F4. The escape's own comment said it "falls back to the ULP floor alone
    — the same thing the `ctl is None` path does", while the code returned
    `max(u, c)`. So a control declared a NON-MEASUREMENT still widened the floor
    whenever it was the larger of the two, and the detail line still printed
    `floor=shape+ulp`, leaving no observable that anything had been declared
    unusable. That is F3's mechanism, one level down.

    Here the control's TV (3.1e-5) is under `_FLOOR_DEGENERATE_TV` and yet 20x
    the ULP estimate (1.6e-6). Taking the max would divide every WDL arm by 20x
    too much; returning the ULP estimates alone does not. The test asserts the
    ARM VALUE, not just the verdict — a verdict-only assertion cannot tell a
    20x-wider floor from a correct one.
    """
    rng = np.random.default_rng(7)
    ref_w = rng.normal(0, 1e-3, size=(32, 3)).astype(np.float32)
    ref_p = rng.normal(0, 4.0, size=(32, 64)).astype(np.float32)
    ctl_w = (ref_w + rng.normal(0, 1e-4, size=ref_w.shape)).astype(np.float32)
    ctl_p = _ulp_scale_control(ref_p, rng)

    ulp = mean_row_tv(_probs(bf16_ulp_perturbation(ref_w, seed=0)), _probs(ref_w))
    ctl_tv = mean_row_tv(_probs(ctl_w), _probs(ref_w))
    assert ctl_tv <= MOD._FLOOR_DEGENERATE_TV, "premise: the control is degenerate"
    assert ctl_tv > 5.0 * ulp, (
        f"premise: the degenerate control ({ctl_tv:.3e}) must be LARGER than the "
        f"ULP estimate ({ulp:.3e}), or `max` and `ulp-only` agree and the "
        "scenario cannot separate them"
    )

    # ⚑ THE NUMERATOR HAS TO SIT BETWEEN THE TWO FLOORS, or the test cannot
    # separate `ulp-only` from `max(ulp, ctl)` — a discrepancy far above BOTH
    # fails either way, and that version of this test let the mutant through.
    aot_w = (ref_w + rng.normal(0, 4e-5, size=ref_w.shape)).astype(np.float32)
    num = mean_row_tv(_probs(aot_w), _probs(ref_w))
    assert 2.0 * ulp < num < 2.0 * ctl_tv, (
        f"premise: the WDL discrepancy ({num:.3e}) must FAIL against the honest "
        f"floor ({ulp:.3e}) and PASS against the inflated one ({ctl_tv:.3e}), or "
        "both behaviours give the same verdict"
    )

    ok, detail, _, _ = _gate(_ulp_scale_control(ref_p, rng), aot_w, ref_p, ref_w,
                             ctl_p, ctl_w)
    arms = _arms(detail)
    assert "wdl:ulp-only(ctl-degenerate)" in detail, (
        f"a control declared a non-measurement is still labelled a contributor: "
        f"{detail}"
    )
    assert arms["wdl_mean"] > 2.0, (
        f"wdl_mean reads x{arms['wdl_mean']:.2f}: the degenerate control widened "
        f"the floor anyway, which is exactly what the old comment denied: {detail}"
    )
    assert not ok, detail


def test_an_IMPLAUSIBLE_floor_fails_the_bucket_without_any_cross_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ F3(b). The absolute bound, and the case it exists for.

    No ratio between the two estimates can catch a floor that is simply too big:
    if BOTH are inflated the same way the cross-check sees perfect agreement,
    and on CPU (`ctl` bitwise identical) or at bucket 1 (`ctl is None`) there is
    no second estimate at all. A row TV is bounded by 1, so a floor of 0.97 —
    what the sentinel-mass defect produced — means the floor claims the two
    distributions are nearly disjoint. That is not bf16 noise at any magnitude.

    This reproduces that defect directly, at `ctl is None`, by substituting a
    perturbation that saturates the softmax. `_FLOOR_IMPLAUSIBLE_TV` alone must
    fail the bucket, with no control involved.
    """
    rng = np.random.default_rng(19)
    ref = (rng.normal(0, 1.0, size=(64, 128)) * 4.0).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(64, 3)).astype(np.float32)

    def _saturating(ref_logits: np.ndarray, *, seed: int = 0) -> np.ndarray:
        """The sentinel-mass defect's signature: a floor near 1.0."""
        g = np.random.default_rng(int(seed))
        return (ref_logits + g.normal(0, 60.0, size=ref_logits.shape)).astype(np.float32)

    monkeypatch.setattr(MOD, "bf16_ulp_perturbation", _saturating)
    aot = _ulp_scale_control(ref, rng)   # as healthy as a package can be
    aot_w = _ulp_scale_control(ref_w, rng)
    ok, detail, _, _ = _gate(aot, aot_w, ref, ref_w)
    arms = _arms(detail)

    assert "FLOOR-IMPLAUSIBLE" in detail, (
        f"a floor near 1.0 was accepted as bf16 noise: {detail}"
    )
    assert "floor=pol:ulp-only wdl:ulp-only" in detail, (
        f"premise: no control here, so the cross-check cannot be what caught "
        f"it: {detail}"
    )
    for arm in ("pol_mean", "pol_tail", "wdl_mean"):
        assert arms[arm] <= 2.0, (
            f"an inflated floor makes every RATIO arm pass — that is the whole "
            f"point — so a ratio arm must not be what failed this: {detail}"
        )
    assert arms["pol_rows_over"] == 0, detail
    assert arms["wdl_rows_over"] == 0, detail
    assert not ok, f"an implausible floor was reported and then ignored: {detail}"


def test_a_healthy_production_floor_is_far_under_the_plausibility_bound() -> None:
    """⚑ The paired negative control for the bound above: it must not fire on
    anything the real net produces.

    Measured 2026-08-15 on bt4heads iter100, worst floor over 80 resamples at
    each of buckets {1, 2, 8, 16, 64, 128}: policy tail 1.89e-2, wdl tail
    6.17e-2 — 4.1x under the 0.25 bound. It is also bounded analytically: one
    bf16 ULP moves a logit by at most |z| * 2^-7, so the largest gap change is
    2*max|z|/128 and a softmax's TV under a gap change d is at most d/4. The
    net's measured max|z_live| is 20.1, capping the floor at ~0.079.
    """
    rng = np.random.default_rng(23)
    # The production regime: policy mean -6.08 spread 3.03, wdl mean -8.46
    # spread 0.93, and the same max|z| the real net reaches.
    for n in (8, 64, 512):
        live = (rng.normal(0, 1.0, size=(n, LIVE_LOGITS)) * 3.03 - 6.08).astype(np.float32)
        wdl = (rng.normal(0, 1.0, size=(n, 3)) * 0.93 - 8.46).astype(np.float32)
        for name, arr in (("pol", _sentinel_padded(live)), ("wdl", wdl)):
            p = _probs(arr)
            f = _probs(bf16_ulp_perturbation(arr, seed=0))
            tail = tail_row_tv(f, p)
            assert tail < MOD._FLOOR_IMPLAUSIBLE_TV / 3.0, (
                f"{name} floor tail {tail:.4f} at bucket {n} is within 3x of the "
                f"{MOD._FLOOR_IMPLAUSIBLE_TV} plausibility bound — the bound is "
                "close to failing healthy buckets"
            )


def test_three_equal_wdl_logits_get_a_real_ULP_but_can_still_NULL_OUT() -> None:
    """⚑⚑ F5, and the part of it the F1 fix does NOT cure. Report, do not hide.

    The net emits `[-15.5625, -15.5625, -15.5625]` — bf16 quantisation makes
    three-equal WDL logits common (3 of 384 real rows measured). Under the OLD
    centred perturbation this row centred to all zeros, so the perturbation was
    exactly zero at EVERY seed and the floor row TV was 0 always.

    The raw-ULP fix gives each entry a real 0.0625 nudge, which is asserted
    below and is deterministic. But the row can STILL null out: all three
    entries share a binade, so their ULPs are equal, and the 2-in-8 sign draw
    that moves them the same way is a pure common shift the softmax cannot see.
    Measured over floor seeds 0..5 this row reads 0.0 at seeds 0 and 4 and
    2.7e-2 to 2.8e-2 at the other four; across 384 REAL WDL rows, 12-17% have a
    zero floor at any one seed.

    At bucket >= 2 the batch mean and tail absorb it. At bucket 1 the floor IS
    that row, so `wdl_mean` reads `inf` and a one-ULP-off package FAILS. Bucket
    1 is off the default ladder but `--buckets 1 --allow-incomplete` reaches it.
    Curing it means replacing the single sign draw with something that cannot be
    a null draw, which re-banks every calibration in this file; it is filed, not
    done, and pinned here so it cannot drift silently.
    """
    row = np.array([[-15.5625, -15.5625, -15.5625]], dtype=np.float32)

    # ⚑ Deterministic half: the F1 fix DOES reach this row. |−15.5625| is in
    # [8,16), so one bf16 ULP is 2^(3-7) = 0.0625, on every entry, every seed.
    for seed in range(6):
        delta = np.abs(bf16_ulp_perturbation(row, seed=seed) - row)
        assert np.array_equal(delta, np.full((1, 3), 0.0625, dtype=np.float32)), (
            f"seed {seed}: the perturbation is not one raw ULP: {delta}"
        )

    # ⚑ Stochastic half: whether that nudge SURVIVES the softmax depends on the
    # sign draw. Both outcomes must occur, or the docstring above is wrong.
    ref_p = _sentinel_padded(
        np.random.default_rng(0).normal(0, 3.0, size=(1, LIVE_LOGITS)).astype(np.float32)
    )
    aot_p = _ulp_scale_control(ref_p, np.random.default_rng(1))
    # ⚑ Exactly one ULP off, and NOT a common shift — written out rather than
    # drawn, because a drawn one could itself be a null and then the numerator
    # is zero too and `_ratio(0, 0)` is 0.0, which would test nothing.
    aot_w = row + np.array([[0.0625, -0.0625, 0.0625]], dtype=np.float32)
    seen: dict[bool, str] = {}
    for seed in range(32):
        _, detail, _, _ = _compare_bucket(
            aot_pol=aot_p, aot_wdl=aot_w, ref_pol=ref_p, ref_wdl=row,
            ctl_pol=None, ctl_wdl=None, tv_ratio_max=2.0, floor_seed=seed,
        )
        seen.setdefault(_arms(detail)["wdl_mean"] == float("inf"), detail)

    assert True in seen, (
        "no floor seed in 0..31 nulled this row out — the docstring's 12-17% "
        f"zero-floor rate no longer holds: {seen}"
    )
    assert "wdl_mean=xinf" in seen[True], seen[True]
    assert False in seen, (
        f"EVERY floor seed nulled this row out — the raw-ULP nudge is not "
        f"reaching the softmax at all: {seen}"
    )
    assert _arms(seen[False])["wdl_mean"] < 2.0, (
        f"on a non-null draw a one-ULP-off package must read ~1: {seen[False]}"
    )


def test_a_zero_floor_reads_INF_on_the_ratio_arm_not_zero() -> None:
    """⚑⚑ The M12 kill. `_ratio` returning 0.0 at a zero floor SURVIVED the
    reviewer's mutation harness, and it survived for an instructive reason: at a
    zero floor the exceedance arms threshold at `k * 0`, so they catch the
    nonzero numerator and the VERDICT is unchanged. Every verdict-only test
    stays green while the ratio arms have been silently retired on exactly the
    inputs where they matter most.

    So this asserts the ARM VALUE. The reference is all-zero logits, which the
    perturbation leaves alone by construction (log2(0) is undefined and the
    honest bf16 step there is a subnormal the softmax cannot represent), so the
    floor is exactly 0 with no dependence on the sign draw.
    """
    ref = np.zeros((8, 16), dtype=np.float32)
    ref_w = np.zeros((8, 3), dtype=np.float32)

    assert np.array_equal(bf16_ulp_perturbation(ref_w, seed=0), ref_w), (
        "premise: an exactly-zero logit is left unperturbed, so the floor here "
        "is exactly 0 at every seed"
    )

    aot_w = ref_w.copy()
    aot_w[0, 0] = 1.0
    ok, detail, _, _ = _gate(ref.copy(), aot_w, ref, ref_w)
    arms = _arms(detail)
    assert arms["wdl_mean"] == float("inf"), (
        f"a zero floor with a nonzero numerator must read inf, not "
        f"x{arms['wdl_mean']}: with 0.0 the arm passes and only the exceedance "
        f"count fails, so the ratio arm is dead on these inputs: {detail}"
    )
    assert not ok, detail

    # And the exceedance arm is indeed ALSO firing here — which is precisely why
    # the assertion above has to be on the value.
    assert arms["wdl_rows_over"] == 1, detail


def test_argmax_min_rejects_nan_and_out_of_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⚑ `rate < nan` is False, so `--argmax-min nan` silently retires the
    argmax criterion while still printing a number in the log line — the same
    hole `--tv-ratio-max` had, one arm over. Out-of-range values are rejected
    for the mirror-image reason: `--argmax-min 1.5` is unsatisfiable, so it
    fails every healthy package.
    """
    monkeypatch.setattr(MOD, "build_reference_model", lambda *a, **k: _StubModel())
    monkeypatch.setattr(MOD, "load_model_config", lambda p: _StubModel())
    monkeypatch.setattr(MOD, "load_checkpoint_state_dict", lambda p: {})
    monkeypatch.setattr(MOD, "_require_cuda", lambda action: None)
    for bad in ("nan", "-0.1", "1.5", "inf"):
        rc = MOD.main([
            "--checkpoint", "x.pt", "--verify-only", "--out-dir", str(tmp_path),
            "--buckets", "8", "--max-batch", "8", "--argmax-min", bad,
        ])
        assert rc == 2, f"--argmax-min {bad} was accepted (rc={rc})"


def test_a_NEAR_degenerate_shape_control_does_not_fail_a_healthy_package() -> None:
    """⚑ THE HOLE THE FIRST DIVERGENCE GUARD LEFT: degeneracy is a RANGE.

    The guard's first revision escaped on ``lo > 0.0`` — exact equality, where a
    band was meant. That covers the CPU regime where ``eager_batch_shape_control``
    is BITWISE identical (TV exactly 0), and misses the case one float away: a
    control that only re-associates a couple of reductions produces a TV around
    1e-8. The softmax is mathematically unchanged, but ``hi / lo`` then reads tens
    of thousands and the bucket FAILS — a healthy package rejected by the gate's
    own safety check, which is the original defect wearing the safety check's hat.

    Measured here: near-degenerate control TV ~9.3e-8 against a real ULP floor of
    ~0.0175, so ``_FLOOR_DEGENERATE_TV`` at 1e-4 sits ~1000x clear of BOTH. The
    package must PASS and must not be labelled FLOOR-DIVERGENCE.
    """
    rng = np.random.default_rng(17)
    ref_pol = np.full((64, 4672), -1e9, dtype=np.float32)
    ref_pol[:, :1858] = (rng.normal(0, 1, (64, 1858)) * 5.0).astype(np.float32)
    ref_wdl = (rng.normal(0, 1, (64, 3)) * 1.5).astype(np.float32)

    # Healthy AOT: one bf16 ULP away. Near-degenerate control: a uniform shift,
    # which softmax is invariant to, so it carries no information about the floor.
    aot_pol = bf16_ulp_perturbation(ref_pol, seed=5)
    aot_wdl = bf16_ulp_perturbation(ref_wdl, seed=5)
    ctl_pol = ref_pol + np.float32(1e-3)
    ctl_wdl = ref_wdl + np.float32(1e-3)

    ok, detail, _, _ = _compare_bucket(
        aot_pol=aot_pol, aot_wdl=aot_wdl,
        ref_pol=ref_pol, ref_wdl=ref_wdl,
        ctl_pol=ctl_pol, ctl_wdl=ctl_wdl,
        tv_ratio_max=2.0,
    )
    assert "FLOOR-DIVERGENCE" not in detail, detail
    assert ok, detail
