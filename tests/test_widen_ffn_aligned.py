"""Tests for tile-aligned FFN widening.

The load-bearing test is `test_widening_preserves_the_function_exactly`: the
whole point of zero-initialising ffn.2's new columns is that the widened network
is the same function. Everything else is bookkeeping around that claim.
"""

from __future__ import annotations

import re

import pytest
import torch

from scripts.widen_ffn_aligned import (
    align_up,
    aligned_ffn_mults,
    plan_widths,
    widen_checkpoint,
)


def test_align_up() -> None:
    assert align_up(796, 128) == 896
    assert align_up(768, 128) == 768  # already aligned -> unchanged
    assert align_up(1, 128) == 128


def test_align_up_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="align must be positive"):
        align_up(10, 0)


def test_aligned_mults_reproduce_their_widths_exactly() -> None:
    """A mult one ULP low would build a width one short, silently."""
    cur = (1.5, 1.555556, 1.65, 1.772222, 1.894444, 1.666667, 1.638889, 1.905556)
    new = aligned_ffn_mults(cur, embed_dim=512, align=128)
    for m in new:
        w = int(512 * m)
        assert w % 128 == 0, f"mult {m} -> width {w}, not a multiple of 128"
    assert [int(512 * m) for m in new] == [768, 896, 896, 1024, 1024, 896, 896, 1024]


def test_a_mult_that_cannot_reproduce_its_width_is_refused() -> None:
    """⚑ The exactness guard is REACHABLE, not decorative.

    Mutation M8 (delete the guard) originally SURVIVED, because no test could
    reach it. Brute force settles whether that is because the guard is dead:
    over integer ``e`` in [1, 4096] and ``t`` in [1, 8192] there are
    **1,853,895** pairs where ``int(e * (t/e)) != t``, and ``e=7, t=61`` is the
    smallest. It is dead only in the PRODUCTION domain — for embed_dim in
    {256..1536} and align in {8..256}, every aligned target round-trips exactly
    (verified by exhaustive sweep), which is why the widths we ship are safe.

    The failure it prevents is silent: a mult one ULP low builds a width one
    column short, and that surfaces only as a shape mismatch at load time.
    """
    with pytest.raises(ValueError, match="does not reproduce width"):
        aligned_ffn_mults((1.0,), embed_dim=7, align=61)


def test_the_production_domain_never_trips_the_exactness_guard() -> None:
    """The other half: the guard must not fire on anything we would ship."""
    for embed_dim in (256, 384, 512, 640, 768, 1024, 1536):
        for align in (8, 16, 32, 64, 128, 256):
            mults = tuple(1.0 + 0.05 * i for i in range(16))
            got = aligned_ffn_mults(mults, embed_dim=embed_dim, align=align)
            for m in got:
                assert int(embed_dim * m) % align == 0


def test_aligned_mults_are_idempotent() -> None:
    once = aligned_ffn_mults((1.555556, 1.905556), embed_dim=512, align=128)
    twice = aligned_ffn_mults(once, embed_dim=512, align=128)
    assert once == twice


def test_production_schedule_param_delta() -> None:
    """The +709,632 figure quoted in the ledger, recomputed from the schedule."""
    cur = (
        1.5, 1.5, 1.5, 1.5, 1.5, 1.555556, 1.65, 1.772222,
        1.894444, 1.666667, 1.638889, 1.905556, 1.783333, 1.794444, 1.744444, 1.5,
    )
    new = aligned_ffn_mults(cur, embed_dim=512, align=128)
    delta = sum((int(512 * n) - int(512 * c)) * 512 * 2 for c, n in zip(cur, new))
    assert delta == 709_632


def _tiny_ckpt(*, mults: tuple[float, ...], embed_dim: int = 8) -> dict:
    """A checkpoint-shaped dict with real ffn tensors and matching opt state."""
    model: dict[str, torch.Tensor] = {}
    names: list[str] = []
    state: dict[int, dict[str, torch.Tensor]] = {}
    g = torch.Generator().manual_seed(7)
    for i, m in enumerate(mults):
        h = int(embed_dim * m)
        for key, shape in (
            (f"blocks.{i}.ffn.0.weight", (h, embed_dim)),
            (f"blocks.{i}.ffn.0.bias", (h,)),
            (f"blocks.{i}.ffn.2.weight", (embed_dim, h)),
            (f"blocks.{i}.ffn.2.bias", (embed_dim,)),
        ):
            model[key] = torch.randn(shape, generator=g)
            names.append(key)
            state[len(names) - 1] = {"momentum_buffer": torch.randn(shape, generator=g)}
    return {
        "model": model,
        "opt": {"state": state, "param_groups": [{"params": list(range(len(names)))}]},
        "opt_param_names": names,
        "arch": {
            "embed_dim": embed_dim,
            "num_layers": len(mults),
            "ffn_mult": mults[0],
            "ffn_mult_by_layer": mults,
        },
    }


def _ffn_forward(model: dict[str, torch.Tensor], layer: int, x: torch.Tensor) -> torch.Tensor:
    w1 = model[f"blocks.{layer}.ffn.0.weight"]
    b1 = model[f"blocks.{layer}.ffn.0.bias"]
    w2 = model[f"blocks.{layer}.ffn.2.weight"]
    b2 = model[f"blocks.{layer}.ffn.2.bias"]
    return torch.nn.functional.mish(x @ w1.T + b1) @ w2.T + b2


def test_widening_preserves_the_function() -> None:
    """THE test: zero-init ffn.2 columns => algebraically identical output.

    ⚑ NOT bitwise. Widening changes the GEMM's N dimension, which changes BLAS
    blocking and therefore float summation ORDER, so even the UNTOUCHED units'
    activations move by ~2 ULP. Measured: max relative diff 2.3e-7 on float32
    (epsilon ~1.2e-7). Padding with literal zero activations reproduces the same
    drift, which is how we know it is reassociation and not the new units.

    The tolerance is therefore real, and it is exactly why
    `test_nonzero_ffn2_columns_break_the_function` exists next to this: a loose
    tolerance would otherwise let a broken (non-zeroed) widening pass.
    """
    mults = (1.25, 1.75)  # embed_dim 8 -> widths 10 and 14, align 8 -> 16 and 16
    ck = _tiny_ckpt(mults=mults)
    before = {k: v.clone() for k, v in ck["model"].items()}
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(11))
    ref = [_ffn_forward(before, i, x) for i in range(len(mults))]

    ck, changes = widen_checkpoint(ck, align=8, seed=3)
    assert changes, "expected some layer to widen"

    for i in range(len(mults)):
        got = _ffn_forward(ck["model"], i, x)
        scale = float(ref[i].abs().max())
        rel = float((got - ref[i]).abs().max()) / max(scale, 1e-12)
        assert rel < 1e-5, f"layer {i} function changed by widening (rel={rel:.3e})"


def test_nonzero_ffn2_columns_break_the_function() -> None:
    """Negative control: the tolerance above must NOT absorb a real error.

    If `test_widening_preserves_the_function` still passed with randomly
    initialised ffn.2 columns, it would be testing nothing.
    """
    ck = _tiny_ckpt(mults=(1.25,))
    before = {k: v.clone() for k, v in ck["model"].items()}
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(11))
    ref = _ffn_forward(before, 0, x)

    old_h = int(8 * 1.25)
    ck, _ = widen_checkpoint(ck, align=8, seed=3)
    w2 = ck["model"]["blocks.0.ffn.2.weight"]
    w2[:, old_h:] = torch.randn_like(w2[:, old_h:])  # sabotage the zeroing

    got = _ffn_forward(ck["model"], 0, x)
    rel = float((got - ref).abs().max()) / float(ref.abs().max())
    assert rel > 1e-2, (
        f"sabotaged widening only moved the output by rel={rel:.3e}; the "
        "preservation test's tolerance cannot distinguish right from wrong"
    )


def test_new_ffn0_rows_are_nonzero_so_gradients_can_flow() -> None:
    """All-zero new rows would deadlock: mish(0)=0 kills the W2 gradient too."""
    ck = _tiny_ckpt(mults=(1.25,))
    old_h = int(8 * 1.25)
    ck, _ = widen_checkpoint(ck, align=8, seed=3)
    new_rows = ck["model"]["blocks.0.ffn.0.weight"][old_h:]
    assert new_rows.numel() > 0
    assert not torch.all(new_rows == 0), "new ffn.0 rows must not be zero"


def test_new_ffn2_columns_are_exactly_zero() -> None:
    ck = _tiny_ckpt(mults=(1.25,))
    old_h = int(8 * 1.25)
    ck, _ = widen_checkpoint(ck, align=8, seed=3)
    new_cols = ck["model"]["blocks.0.ffn.2.weight"][:, old_h:]
    assert new_cols.numel() > 0
    assert torch.all(new_cols == 0), "ffn.2 new columns must be exactly zero"


def test_optimizer_momentum_is_migrated_not_dropped() -> None:
    """Shapes must track the parameter, and the ORIGINAL moments must survive."""
    ck = _tiny_ckpt(mults=(1.25,))
    names = list(ck["opt_param_names"])
    old = {
        n: ck["opt"]["state"][names.index(n)]["momentum_buffer"].clone()
        for n in names
        if "ffn" in n
    }
    old_h = int(8 * 1.25)
    ck, _ = widen_checkpoint(ck, align=8, seed=3)
    for n, prev in old.items():
        buf = ck["opt"]["state"][names.index(n)]["momentum_buffer"]
        param = ck["model"][n]
        assert buf.shape == param.shape, f"{n}: moment {buf.shape} != param {param.shape}"
        if n.endswith("ffn.2.weight"):
            assert torch.equal(buf[:, :old_h], prev), f"{n}: original moments lost"
            assert torch.all(buf[:, old_h:] == 0)
        elif n.endswith("ffn.0.weight") or n.endswith("ffn.0.bias"):
            assert torch.equal(buf[:old_h], prev), f"{n}: original moments lost"
            assert torch.all(buf[old_h:] == 0)


def test_arch_is_rewritten_or_resume_rebuilds_the_old_widths() -> None:
    ck = _tiny_ckpt(mults=(1.25, 1.75))
    ck, _ = widen_checkpoint(ck, align=8, seed=3)
    for m in ck["arch"]["ffn_mult_by_layer"]:
        assert int(8 * m) % 8 == 0
    assert [int(8 * m) for m in ck["arch"]["ffn_mult_by_layer"]] == [16, 16]


def test_already_aligned_checkpoint_is_a_noop() -> None:
    ck = _tiny_ckpt(mults=(2.0,))  # width 16, align 8 -> unchanged
    before = {k: v.clone() for k, v in ck["model"].items()}
    ck, changes = widen_checkpoint(ck, align=8, seed=3)
    assert changes == {}
    for k, v in before.items():
        assert torch.equal(ck["model"][k], v)


def test_plan_widths_reports_only_growing_layers() -> None:
    ck = _tiny_ckpt(mults=(2.0, 1.25))
    changes, _ = plan_widths(ck["arch"], align=8)
    assert 0 not in changes           # already 16, aligned
    assert changes[1] == (10, 16)


def test_widen_refuses_a_checkpoint_without_arch() -> None:
    with pytest.raises(ValueError, match="no 'arch'"):
        widen_checkpoint({"model": {}}, align=128)


def test_module_docstring_states_the_zero_column_reason() -> None:
    """The 'why' must survive refactors; it is the non-obvious half."""
    import scripts.widen_ffn_aligned as mod

    assert mod.__doc__ is not None
    assert re.search(r"zero", mod.__doc__, re.I)
    assert "mish(0) = 0" in mod.__doc__ or "mish(0)" in mod.__doc__
