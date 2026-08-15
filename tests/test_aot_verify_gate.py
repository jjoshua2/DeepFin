"""Tests for the self-calibrating AOT verify gate.

The load-bearing test is `test_gate_verdict_is_invariant_to_policy_sharpness`:
that is the exact failure that broke the old gate on 2026-08-15, when the same
deployed packages went PASS -> FAIL purely because the net's top-1 probability
grew 4.4x. A gate whose verdict moves with sharpness is not measuring the
packages.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.build_aot_packages import (
    _compare_bucket,
    eager_batch_shape_control,
    mean_row_tv,
)


def _probs(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def test_mean_row_tv_bounds() -> None:
    p = np.array([[0.5, 0.5], [0.25, 0.75]])
    assert mean_row_tv(p, p) == pytest.approx(0.0)
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert mean_row_tv(a, b) == pytest.approx(1.0)  # disjoint support


def test_mean_row_tv_averages_rows_not_maxes_them() -> None:
    """A max would report 1.0 here; a mean reports 0.5. That is the whole fix."""
    a = np.array([[1.0, 0.0], [0.5, 0.5]])
    b = np.array([[0.0, 1.0], [0.5, 0.5]])
    assert mean_row_tv(a, b) == pytest.approx(0.5)


def _case(*, aot_off: float, ctl_off: float, n: int = 64, sharp: float = 1.0):
    """Build (aot, ref, ctl) logit triples with controlled perturbations."""
    rng = np.random.default_rng(5)
    ref = rng.normal(0.0, 1.0, size=(n, 40)) * sharp
    # bf16 error is RELATIVE: absolute logit error scales with logit magnitude.
    # Holding it fixed while scaling `ref` would model shrinking relative
    # precision, which is not what sharpening does.
    aot = ref + rng.normal(0.0, aot_off * sharp, size=ref.shape)
    ctl = ref + rng.normal(0.0, ctl_off * sharp, size=ref.shape)
    wdl_ref = rng.normal(0.0, 1.0, size=(n, 3)) * sharp
    return (
        aot, wdl_ref + rng.normal(0.0, aot_off * sharp, size=(n, 3)),
        ref, wdl_ref,
        ctl, wdl_ref + rng.normal(0.0, ctl_off * sharp, size=(n, 3)),
    )


def test_gate_passes_when_aot_matches_the_control_level() -> None:
    a_p, a_w, r_p, r_w, c_p, c_w = _case(aot_off=0.02, ctl_off=0.02)
    ok, detail = _compare_bucket(
        aot_pol=a_p, aot_wdl=a_w, ref_pol=r_p, ref_wdl=r_w,
        ctl_pol=c_p, ctl_wdl=c_w, tv_ratio_max=1.5, argmax_min=0.5,
    )
    assert ok, detail


def test_gate_fails_when_aot_is_much_worse_than_the_control() -> None:
    a_p, a_w, r_p, r_w, c_p, c_w = _case(aot_off=0.40, ctl_off=0.02)
    ok, detail = _compare_bucket(
        aot_pol=a_p, aot_wdl=a_w, ref_pol=r_p, ref_wdl=r_w,
        ctl_pol=c_p, ctl_wdl=c_w, tv_ratio_max=1.5, argmax_min=0.0,
    )
    assert not ok, detail


def test_gate_verdict_is_invariant_to_policy_sharpness() -> None:
    """⚑ THE regression test for the 2026-08-15 failure.

    The old gate was `max |p_aot - p_ref| <= 0.02`. Scaling the logits leaves
    the RELATIVE agreement untouched but concentrates the probabilities, which
    is what flipped identical deployed packages from PASS to FAIL. The ratio
    gate must return the same verdict at both sharpness levels.
    """
    verdicts = []
    old_gate_maxdev = []
    for sharp in (1.0, 6.0):
        a_p, a_w, r_p, r_w, c_p, c_w = _case(aot_off=0.02, ctl_off=0.02, sharp=sharp)
        ok, _ = _compare_bucket(
            aot_pol=a_p, aot_wdl=a_w, ref_pol=r_p, ref_wdl=r_w,
            ctl_pol=c_p, ctl_wdl=c_w, tv_ratio_max=1.5, argmax_min=0.0,
        )
        verdicts.append(ok)
        old_gate_maxdev.append(float(np.max(np.abs(_probs(a_p) - _probs(r_p)))))

    assert verdicts[0] == verdicts[1], (
        f"gate verdict moved with sharpness alone: {verdicts}"
    )
    # And confirm the OLD statistic really would have moved — otherwise this
    # test proves nothing about the defect it claims to guard.
    assert old_gate_maxdev[1] > 3 * old_gate_maxdev[0], (
        f"sharpening did not move the old max-dev statistic "
        f"({old_gate_maxdev}); the scenario does not reproduce the defect"
    )


def test_gate_is_scale_free_in_the_noise_floor() -> None:
    """⚑ The gate must depend on the RATIO, never on absolute TV.

    Two scenarios with the SAME aot/control ratio (~1) but very different
    absolute deviation. A correct ratio gate passes both. Any absolute
    threshold — including a ratio whose denominator has been frozen to a
    constant — passes the quiet one and fails the noisy one.

    This is the mutant `test_gate_verdict_is_invariant_to_policy_sharpness`
    did NOT kill: substituting `tv_pol / 0.02` for the measured control left
    that test green, because it held aot_off == ctl_off at a single noise
    level. Sharpness invariance and scale-freedom are different properties.
    """
    verdicts = []
    tvs = []
    for off in (0.02, 0.5):
        a_p, a_w, r_p, r_w, c_p, c_w = _case(aot_off=off, ctl_off=off)
        ok, _ = _compare_bucket(
            aot_pol=a_p, aot_wdl=a_w, ref_pol=r_p, ref_wdl=r_w,
            ctl_pol=c_p, ctl_wdl=c_w, tv_ratio_max=1.5, argmax_min=0.0,
        )
        verdicts.append(ok)
        tvs.append(mean_row_tv(_probs(a_p), _probs(r_p)))

    assert all(verdicts), (
        f"a scale-free gate must pass both noise levels, got {verdicts} "
        f"at TV {tvs} — the denominator is not tracking the control"
    )
    # The scenario is only meaningful if the two TVs really are far apart.
    assert tvs[1] > 5 * tvs[0], (
        f"noise levels too close ({tvs}); scenario cannot separate a ratio "
        "gate from an absolute one"
    )


def test_absent_control_degrades_to_argmax_and_says_so() -> None:
    a_p, a_w, r_p, r_w, _, _ = _case(aot_off=0.02, ctl_off=0.02)
    ok, detail = _compare_bucket(
        aot_pol=a_p, aot_wdl=a_w, ref_pol=r_p, ref_wdl=r_w,
        ctl_pol=None, ctl_wdl=None, tv_ratio_max=1.5, argmax_min=0.0,
    )
    assert ok
    assert "no control" in detail, "a weakened gate must announce itself"


def test_argmax_floor_still_rejects_garbage() -> None:
    """A package with unfilled constants collapses argmax toward random."""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(128, 40))
    garbage = rng.normal(size=(128, 40))  # independent => argmax agrees ~1/40
    ok, detail = _compare_bucket(
        aot_pol=garbage, aot_wdl=rng.normal(size=(128, 3)),
        ref_pol=ref, ref_wdl=rng.normal(size=(128, 3)),
        ctl_pol=ref, ctl_wdl=rng.normal(size=(128, 3)),
        tv_ratio_max=1e9, argmax_min=0.90,  # ratio disabled: argmax must catch it
    )
    assert not ok, detail


def test_eager_control_returns_none_when_batch_cannot_be_rechunked() -> None:
    model = torch.nn.Identity()
    pol, wdl = eager_batch_shape_control(model, torch.zeros(1, 3))
    assert pol is None and wdl is None


def test_eager_control_uses_a_different_batch_shape() -> None:
    """The control is worthless if it re-runs at the SAME shape."""
    seen: list[int] = []

    class _Recorder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            seen.append(int(x.shape[0]))
            n = int(x.shape[0])
            return {"policy": torch.zeros(n, 1858), "wdl": torch.zeros(n, 3)}

    eager_batch_shape_control(_Recorder(), torch.zeros(16, 3))
    assert seen, "control never ran the model"
    assert all(s != 16 for s in seen), f"control re-ran at the SAME shape: {seen}"
    assert sum(seen) == 16, f"control did not cover every row exactly once: {seen}"
