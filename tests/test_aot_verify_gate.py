"""Tests for the self-calibrating AOT verify gate.

Five properties carry this gate, and they are DIFFERENT properties. A previous
revision tested only the first and shipped two live holes:

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

def test_ulp_floor_is_never_zero_where_the_shape_control_is() -> None:
    """⚑ THE F2 regression test.

    Round-tripping already-bf16 logits through bf16 is a no-op, and the
    batch-shape control is bitwise identical on CPU — both give a floor of
    exactly 0. A +/-1 ULP nudge does not.
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
    scenario has to live inside that window: measured here, ULP-only reads
    pol_mean x2.99 (FAIL at the 2.0 default) and the matched control pulls it to
    x1.05 (PASS) at a floor divergence of 2.9x. The pre-guard version of this
    test used a 0.30 control, which now reads 7.1x and fails for the OTHER
    reason — a test that would have gone green on a floor that never widened.
    """
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 4.0, size=(128, 256)).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(128, 3)).astype(np.float32)
    aot = ref + rng.normal(0, 0.12, size=ref.shape).astype(np.float32)
    aot_w = ref_w.copy()

    tight, tight_detail, _, _ = _gate(aot, aot_w, ref, ref_w)  # ULP floor only
    ctl = ref + rng.normal(0, 0.12, size=ref.shape).astype(np.float32)
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
    # 0.01, not 0.02: on sd-4 logits a 0.02 control reads 7.4x the analytic
    # floor and the bucket then fails as FLOOR-DIVERGENCE, i.e. for a reason
    # that has nothing to do with the row-local damage this test is about.
    ctl = ref + rng.normal(0, 0.01, size=ref.shape).astype(np.float32)
    ctl_w = ref_w + rng.normal(0, 0.01, size=ref_w.shape).astype(np.float32)

    aot = ref + rng.normal(0, 0.01, size=ref.shape).astype(np.float32)
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
    ctl = ref + rng.normal(0, 0.01, size=ref.shape).astype(np.float32)
    ctl_w = ref_w + rng.normal(0, 0.01, size=ref_w.shape).astype(np.float32)

    aot = ref + rng.normal(0, 0.01, size=ref.shape).astype(np.float32)
    hurt = int(n * 0.08)
    aot[:hurt] = ref[:hurt] + rng.normal(0, 0.10, size=(hurt, 128)).astype(np.float32)

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
    """
    rows = _run_verify(tmp_path, monkeypatch, offset=0.0)[2]
    assert "floor=shape+ulp" in rows[0][2], (
        f"the batch-shape control never reached _compare_bucket: {rows}"
    )


def test_the_ulp_floor_magnitude_quoted_in_the_docstring_is_reproducible() -> None:
    """⚑ BANK THE DUMP: the docstring's 0.0202 must be re-derived, not trusted.

    This is the one calibration number in the gate that CI hardware can
    reproduce. The CUDA readings (0.0176 control / 0.01737 AOT at bucket 1190)
    cannot be, and are labelled as provenance rather than calibration in the
    source. If this value drifts, `--tv-ratio-max`'s default needs revisiting.
    """
    rng = np.random.default_rng(0)
    ref = torch.from_numpy(
        (rng.normal(0, 1.0, size=(256, 1858)) * 6.0).astype(np.float32)
    ).to(torch.bfloat16).float().numpy()
    p = _probs(ref)
    top1 = float(p.max(axis=-1).mean())
    floor = mean_row_tv(_probs(bf16_ulp_perturbation(ref, seed=0)), p)

    assert 0.50 < top1 < 0.65, f"sharpness regime moved: top-1 {top1:.4f}"
    assert 0.015 < floor < 0.030, (
        f"ULP floor {floor:.4f} is outside the docstring's ~0.0202; the "
        "tv_ratio_max default was chosen against that magnitude"
    )


def test_the_ulp_floor_is_INVARIANT_to_a_common_logit_shift() -> None:
    """⚑ Otherwise the gate is weight-dependent again — the defect it exists for.

    Adding a constant to every logit leaves the softmax unchanged, and leaves
    the measured AOT-vs-eager TV unchanged too (the same bias tensor rounds
    identically in both pipelines, so it cancels in their difference). A floor
    derived from the RAW logits does NOT stay put: measured 0.0194 -> 0.0310 at
    +10 and -> 0.1329 at +100. A head bias drifting common-mode would then widen
    the denominator until the same package discrepancy started passing.
    """
    rng = np.random.default_rng(0)
    base = torch.from_numpy(
        (rng.normal(0, 1.0, size=(256, 512)) * 6.0).astype(np.float32)
    ).to(torch.bfloat16).float().numpy()

    floors = [
        mean_row_tv(_probs(bf16_ulp_perturbation(base + off, seed=0)),
                    _probs(base + off))
        for off in (0.0, 10.0, 100.0)
    ]
    assert max(floors) / min(floors) < 1.10, (
        f"floor moved with an arbitrary common offset: {floors}"
    )
    # And the scenario must actually be one where a RAW floor would move,
    # or this test proves nothing.
    raw = []
    for off in (0.0, 100.0):
        z = (base + off).astype(np.float32)
        s = np.random.default_rng(0).choice(
            np.array([-1.0, 1.0], dtype=np.float32), size=z.shape)
        raw.append(mean_row_tv(_probs(z * (1.0 + s * np.float32(2.0) ** -8)), _probs(z)))
    assert raw[1] > 3 * raw[0], f"scenario does not reproduce the defect: {raw}"


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
    Healthy is modelled as the floor's OWN functional at a different seed: two
    bf16 pipelines that disagree by one ULP is precisely what "healthy" means
    here, and it is what the source's banked table was measured against.

    MEASURED 2026-08-15, buckets {8, 64, 512} x 8 seeds, n=24 per arm:

        arm            mean    median   worst
        pol_mean       1.036   1.005    1.410
        pol_tail       1.011   1.000    1.160
        wdl_mean       0.924   0.965    1.060
        pol_rows_over  0       0        0
        wdl_rows_over  0       0        0

    The worst reading is 1.410 against a 2.0 default, and it comes from bucket
    8 — 8 rows is where every one of these estimators is noisiest, which is why
    the sweep includes it. A wider sweep (buckets {8,32,128,512,1024} x 12
    seeds, n=60) reads 1.009/0.997/0.963 with worst 1.380, matching the table in
    the source's calibration block (1.003/0.990/0.989, worst 1.462).

    If this drifts, --tv-ratio-max's default needs revisiting.
    """
    seen: dict[str, list[float]] = {}
    for n in (8, 64, 512):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            live = (rng.normal(0, 1.0, size=(n, LIVE_LOGITS)) * 5.0).astype(np.float32)
            ref_w = rng.normal(0, 1.0, size=(n, 3)).astype(np.float32)
            _, detail, _, _ = _compare_bucket(
                aot_pol=_sentinel_padded(bf16_ulp_perturbation(live, seed=1000 + seed)),
                aot_wdl=bf16_ulp_perturbation(ref_w, seed=2000 + seed),
                ref_pol=_sentinel_padded(live), ref_wdl=ref_w,
                ctl_pol=None, ctl_wdl=None, tv_ratio_max=1e12,
            )
            for arm, val in _arms(detail).items():
                seen.setdefault(arm, []).append(val)

    assert set(seen) == {
        "pol_mean", "pol_tail", "wdl_mean", "pol_rows_over", "wdl_rows_over",
    }, f"arm set drifted: {sorted(seen)}"

    for arm in ("pol_mean", "pol_tail", "wdl_mean"):
        vals = seen[arm]
        mean = float(np.mean(vals))
        assert 0.8 < mean < 1.2, (
            f"arm {arm} reads {mean:.3f} on a HEALTHY production-shaped package, "
            "not ~1.0 — one --tv-ratio-max no longer means the same thing on "
            "every arm, which is what deleting _ARM_NULL asserted"
        )
        assert max(vals) < 1.7, (
            f"arm {arm} peaked at {max(vals):.2f} on a HEALTHY package against a "
            "2.0 default bound — the gate is close to false-positiving"
        )

    # The count arms are gated at ZERO, so their null is not "about one", it is
    # "never". A single false exceedance here would make the gate reject good
    # packages outright, with no threshold left to absorb it.
    for arm in ("pol_rows_over", "wdl_rows_over"):
        assert max(seen[arm]) == 0, (
            f"{arm} fired on a healthy package: {seen[arm]} — _ROW_EXCEED_K "
            f"= {MOD._ROW_EXCEED_K} is too tight"
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


def test_the_floor_is_shift_invariant_at_the_production_width() -> None:
    """⚑ Shift-invariance has to survive the mask, not just the narrow case.

    Softmax ignores a common offset, so the floor must too — otherwise a head
    bias drifting common-mode widens the denominator and the same package
    discrepancy starts passing. The live-entry mean is the only shift-invariant
    centre available once 60% of the row is a sentinel that does NOT move with
    the offset: a whole-row mean shifts by 1858/4672 of it, i.e. by a different
    amount than the logits did, and shift-invariance is lost.
    """
    rng = np.random.default_rng(2)
    padded = _sentinel_padded(
        (rng.normal(0, 1.0, size=(128, LIVE_LOGITS)) * 5.0).astype(np.float32)
    )
    shifted = padded.copy()
    shifted[:, :LIVE_LOGITS] += 500.0  # sentinels deliberately left where they are

    base = mean_row_tv(_probs(bf16_ulp_perturbation(padded, seed=0)), _probs(padded))
    moved = mean_row_tv(_probs(bf16_ulp_perturbation(shifted, seed=0)), _probs(shifted))

    assert base > 0.0, "premise: the floor is non-degenerate"
    assert moved == pytest.approx(base, rel=1e-3), (
        f"the floor moved with a common shift of the LIVE logits: {base:.6f} -> "
        f"{moved:.6f}. The softmax did not move, so the floor must not either."
    )


def test_sentinel_slots_stay_dead_after_perturbation() -> None:
    """⚑ The masked slots must carry exactly zero probability, still.

    The fix perturbs only the live entries; the sentinels are shifted by the
    live mean and otherwise left alone. If a perturbation ever reached them —
    or if the shift lifted them into range — mass would leak into illegal moves
    and the floor would be measuring the mask rather than the policy.

    ⚑ AND THE CONTRACT HAS TO BE ASSERTED ON THE LOGITS, NOT ONLY ON THE
    PROBABILITIES. Measured 2026-08-15: perturbing the sentinels too (dropping
    the `np.where` and returning `z * (1 + sign*step)` wholesale) leaves the
    softmax BIT-IDENTICAL and the sentinel mass at exactly 0, because -1e9
    scaled by 1 +/- 2^-8 is -1e9 +/- 3.9e6 and `exp` of either is 0. So a
    probability-space assertion alone cannot distinguish the two — that mutant
    SURVIVED the first round of this test. It is harmless only because the
    sentinel is nine orders of magnitude below the live range; the moment that
    margin shrinks it stops being harmless, which is exactly what a contract
    assertion is for.
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
    assert p.sum(axis=-1) == pytest.approx(np.ones(64), rel=0, abs=1e-9), (
        "rows no longer normalize to 1 after the perturbation"
    )

    # Premise for the contract check: every live entry is inside the 80-logit
    # window, so `live` selects exactly the 1858 real columns and the centre is
    # the live mean.
    assert float(np.max(live.max(axis=-1) - live.min(axis=-1))) < 80.0
    expected_sentinel = np.float32(SENTINEL) - live.mean(axis=-1, keepdims=True)
    assert np.allclose(
        perturbed[:, LIVE_LOGITS:], expected_sentinel, rtol=1e-6, atol=0.0
    ), (
        "the sentinel slots were PERTURBED, not merely shifted; a 2^-8 relative "
        "nudge of -1e9 is +/-3.9e6, which the softmax cannot see today and will "
        "not always be able to ignore"
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
