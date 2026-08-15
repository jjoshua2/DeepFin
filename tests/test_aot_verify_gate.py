"""Tests for the self-calibrating AOT verify gate.

Three properties carry this gate, and they are DIFFERENT properties. A previous
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

Plus the standing rule for this repo: every knob must be shown to REACH the
comparison, and every branch must be shown to be able to FAIL. A gate that
cannot fail is the defect this codebase produces most often.
"""

from __future__ import annotations

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


def _probs(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


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
    """A large empirical control must WIDEN the gate relative to ULP alone."""
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 4.0, size=(128, 256)).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(128, 3)).astype(np.float32)
    aot = ref + rng.normal(0, 0.30, size=ref.shape).astype(np.float32)
    aot_w = ref_w.copy()

    tight, _, _, _ = _gate(aot, aot_w, ref, ref_w)  # ULP floor only
    ctl = ref + rng.normal(0, 0.30, size=ref.shape).astype(np.float32)
    loose, _, _, _ = _gate(aot, aot_w, ref, ref_w, ctl, ref_w.copy())
    assert not tight, "the ULP-only floor should have been too tight here"
    assert loose, (
        "an empirical control at the SAME noise level as the package must make "
        "the gate pass; it did not widen the floor"
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
    ok, detail, _, _ = _gate(*_case(aot_off=0.02, ctl_off=0.02))
    assert ok, detail


def test_gate_fails_when_aot_is_much_worse_than_the_control() -> None:
    ok, detail, _, _ = _gate(*_case(aot_off=0.60, ctl_off=0.02))
    assert not ok, detail


def test_gate_verdict_is_invariant_to_policy_sharpness() -> None:
    """⚑ Property 1 — THE regression test for the 2026-08-15 failure."""
    verdicts, old_gate_maxdev = [], []
    for sharp in (1.0, 6.0):
        a_p, a_w, r_p, r_w, c_p, c_w = _case(aot_off=0.02, ctl_off=0.02, sharp=sharp)
        ok, _, _, _ = _gate(a_p, a_w, r_p, r_w, c_p, c_w)
        verdicts.append(ok)
        old_gate_maxdev.append(float(np.max(np.abs(_probs(a_p) - _probs(r_p)))))

    assert verdicts[0] == verdicts[1], f"verdict moved with sharpness alone: {verdicts}"
    # And confirm the OLD statistic really would have moved — otherwise this
    # test proves nothing about the defect it claims to guard.
    assert old_gate_maxdev[1] > 3 * old_gate_maxdev[0], (
        f"sharpening did not move the old max-dev statistic ({old_gate_maxdev}); "
        "the scenario does not reproduce the defect"
    )


def test_gate_is_scale_free_in_the_noise_floor() -> None:
    """⚑ Property 2 — kills a frozen denominator, which property 1 does not."""
    verdicts, tvs = [], []
    for off in (0.02, 0.5):
        a_p, a_w, r_p, r_w, c_p, c_w = _case(aot_off=off, ctl_off=off)
        ok, _, _, _ = _gate(a_p, a_w, r_p, r_w, c_p, c_w)
        verdicts.append(ok)
        tvs.append(mean_row_tv(_probs(a_p), _probs(r_p)))
    assert all(verdicts), f"a scale-free gate must pass both noise levels: {verdicts} at TV {tvs}"
    assert tvs[1] > 5 * tvs[0], f"noise levels too close ({tvs}) to separate the designs"


def test_row_local_damage_fails_even_though_the_mean_is_clean() -> None:
    """⚑ The coverage the old whole-array max had, that a mean alone loses.

    A kernel wrong on 2% of rows — a boundary/indexing defect at one bucket
    size — barely moves the mean. The tail arm is what catches it.
    """
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 4.0, size=(500, 128)).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(500, 3)).astype(np.float32)
    ctl = ref + rng.normal(0, 0.02, size=ref.shape).astype(np.float32)
    ctl_w = ref_w + rng.normal(0, 0.02, size=ref_w.shape).astype(np.float32)

    aot = ref + rng.normal(0, 0.02, size=ref.shape).astype(np.float32)
    aot[:10] = rng.normal(0, 4.0, size=(10, 128))  # 2% of rows: garbage

    ok, detail, _, _ = _gate(aot, ctl_w.copy(), ref, ref_w, ctl, ctl_w)
    assert not ok, f"row-local corruption passed the gate: {detail}"


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
    """The knob must REACH the comparison, not merely be accepted."""
    args = _case(aot_off=0.30, ctl_off=0.02)
    assert not _gate(*args, tv_ratio_max=1.5)[0]
    assert _gate(*args, tv_ratio_max=1e9)[0]


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
    wide = _run_verify(tmp_path, monkeypatch, offset=4.0, argmax_min=0.0,
                       tv_ratio_max=1e12)
    tight = _run_verify(tmp_path, monkeypatch, offset=4.0, argmax_min=0.0,
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
    the tail crosses it: ~1.5% of rows perturbed moderately, which is exactly
    the signature of a boundary defect at one bucket size.
    """
    rng = np.random.default_rng(11)
    n = 2000
    ref = rng.normal(0, 4.0, size=(n, 128)).astype(np.float32)
    ref_w = rng.normal(0, 1.0, size=(n, 3)).astype(np.float32)
    ctl = ref + rng.normal(0, 0.05, size=ref.shape).astype(np.float32)
    ctl_w = ref_w + rng.normal(0, 0.05, size=ref_w.shape).astype(np.float32)

    aot = ref + rng.normal(0, 0.05, size=ref.shape).astype(np.float32)
    hurt = int(n * 0.015)
    aot[:hurt] = ref[:hurt] + rng.normal(0, 1.2, size=(hurt, 128)).astype(np.float32)

    mean_ratio = (mean_row_tv(_probs(aot), _probs(ref))
                  / mean_row_tv(_probs(ctl), _probs(ref)))
    assert mean_ratio < 1.5, (
        f"scenario is not tail-only: the mean arm already fires at x{mean_ratio:.2f}, "
        "so this test would pass with the tail arm deleted"
    )
    ok, detail, _, _ = _gate(aot, ctl_w.copy(), ref, ref_w, ctl, ctl_w)
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

    The worst-row arm is a RATIO OF TWO MAXIMA over the same N, so the
    extreme-value growth that broke the original gate cancels instead of
    accumulating.
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

    A max over 512 rows is naturally 3-5x the mean of the same rows. So if the
    worst-row arm divided by the floor's MEAN instead of the floor's MAX, this
    perfectly healthy package would read ~4x and FAIL — a gate that rejects
    good packages at big buckets, which is exactly the shape of the original
    defect. Every arm's denominator must be the same statistic as its numerator.
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


def test_the_arm_null_table_is_reproducible_and_every_arm_reads_about_one() -> None:
    """⚑⚑ BANK THE DUMP. _ARM_NULL is a MEASUREMENT, and this re-derives it.

    A single threshold across six arms was wrong and shipped: at 1.5, a
    perfectly healthy package read 1.90 on wdl_tail and 2.07 on wdl_max and
    would have FAILED every time — the original defect (a gate that rejects good
    packages) in a new costume. Numerator and denominator are matched in
    FUNCTIONAL but are different random draws, and a p99 or max over 3 WDL
    columns is far noisier than a mean over 1858.

    If this drifts, --tv-ratio-max's default needs revisiting.
    """
    import re as _re

    from scripts.build_aot_packages import _ARM_NULL

    seen: dict[str, list[float]] = {}
    for n in (64, 512):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            ref = (rng.normal(0, 1.0, size=(n, 1024)) * 5.0).astype(np.float32)
            ref_w = rng.normal(0, 1.0, size=(n, 3)).astype(np.float32)
            noise = np.float32(2.0) ** -8
            aot = ref * (1.0 + rng.normal(0, noise, size=ref.shape).astype(np.float32))
            aot_w = ref_w * (1.0 + rng.normal(0, noise, size=ref_w.shape).astype(np.float32))
            _, detail, _, _ = _compare_bucket(
                aot_pol=aot, aot_wdl=aot_w, ref_pol=ref, ref_wdl=ref_w,
                ctl_pol=None, ctl_wdl=None, tv_ratio_max=1e12,
            )
            for k, v in _re.findall(r"(\w+)=x([\d.]+)", detail):
                seen.setdefault(k, []).append(float(v))

    assert set(seen) == set(_ARM_NULL), f"arm set drifted: {sorted(seen)}"
    for arm, vals in seen.items():
        med = float(np.median(vals))
        assert 0.6 < med < 1.7, (
            f"NORMALIZED arm {arm} reads {med:.2f} when healthy, not ~1.0 — "
            f"_ARM_NULL[{arm}]={_ARM_NULL[arm]} no longer matches the measurement"
        )
        assert max(vals) < 2.0, (
            f"arm {arm} peaked at {max(vals):.2f} on a HEALTHY package, which is "
            "at the default bound — the gate would false-positive"
        )
