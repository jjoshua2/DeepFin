"""Tests for tile-aligned FFN widening.

The load-bearing test is `test_widening_preserves_the_function_exactly`: the
whole point of zero-initialising ffn.2's new columns is that the widened network
is the same function. Everything else is bookkeeping around that claim.
"""

from __future__ import annotations

import math
import re

import pytest
import torch

from scripts.widen_ffn_aligned import (
    _pad_cols,
    _pad_rows,
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
    grow = [int(512 * n) - int(512 * c) for c, n in zip(cur, new)]
    # ⚑ THREE tensors grow per layer, not two: ffn.0.weight (g x 512),
    # ffn.2.weight (512 x g) AND ffn.0.bias (g). An earlier version of this
    # test omitted the bias and asserted 709,632 — which is why the measured
    # delta on a real checkpoint came out 693 higher (693 == sum(grow)).
    assert sum(grow) == 693
    delta = sum(g * 512 * 2 + g for g in grow)
    assert delta == 710_325


def _tiny_ckpt(*, mults: tuple[float, ...], embed_dim: int = 8) -> dict:
    """A checkpoint-shaped dict mirroring PRODUCTION's mixed optimizer state.

    ⚑ Measured on bt4heads armB iter100 (479 params): AuroraWithAuxAdam splits
    on rank, not on module. 2-D weights carry ``momentum_buffer`` (48 params);
    1-D biases go through the AdamW fallback and carry ``exp_avg`` /
    ``exp_avg_sq`` / ``step`` (431 params). An earlier version of this fixture
    gave EVERY parameter a ``momentum_buffer``, which is why the tool shipped
    handling only that buffer and would have crashed on the first resumed step
    in ``exp_avg.add_(grad)``. A fixture that is tidier than production tests a
    system that does not exist.
    """
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
            if len(shape) == 2:  # Aurora matrix path
                state[len(names) - 1] = {
                    "momentum_buffer": torch.randn(shape, generator=g)
                }
            else:  # AdamW fallback path
                state[len(names) - 1] = {
                    "exp_avg": torch.randn(shape, generator=g),
                    "exp_avg_sq": torch.rand(shape, generator=g),
                    "step": torch.tensor(1234.0),
                }
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


def test_EVERY_shape_coupled_optimizer_tensor_is_migrated() -> None:
    """⚑ Not just momentum_buffer — that omission was a first-step crash.

    ``ffn.0.bias`` is widened by this tool AND lives on the AdamW side, so its
    ``exp_avg``/``exp_avg_sq`` must widen with it. The migration is matched by
    SHAPE, not by buffer name, so this asserts over whatever the entry holds.
    """
    ck = _tiny_ckpt(mults=(1.25,))
    names = list(ck["opt_param_names"])
    before = {
        n: {k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in ck["opt"]["state"][names.index(n)].items()}
        for n in names if "ffn" in n
    }
    old_h = int(8 * 1.25)
    ck, _ = widen_checkpoint(ck, align=8, seed=3)

    seen_adamw = False
    for n, prev in before.items():
        entry = ck["opt"]["state"][names.index(n)]
        param = ck["model"][n]
        for key, old_val in prev.items():
            new_val = entry[key]
            if not isinstance(old_val, torch.Tensor) or old_val.ndim == 0:
                assert new_val is old_val or torch.equal(new_val, old_val), (
                    f"{n}.{key}: a scalar like 'step' must not be touched"
                )
                continue
            if key in ("exp_avg", "exp_avg_sq"):
                seen_adamw = True
            if old_val.shape == param.shape:
                continue  # this tensor did not need to grow
            assert new_val.shape == param.shape, (
                f"{n}.{key}: {tuple(new_val.shape)} != param {tuple(param.shape)} "
                "— the first resumed step dies here"
            )
            if n.endswith("ffn.2.weight"):
                assert torch.equal(new_val[:, :old_h], old_val), f"{n}.{key}: history lost"
                assert torch.all(new_val[:, old_h:] == 0)
            else:
                assert torch.equal(new_val[:old_h], old_val), f"{n}.{key}: history lost"
                assert torch.all(new_val[old_h:] == 0)

    assert seen_adamw, (
        "the fixture no longer exercises the AdamW fallback; this test would "
        "pass again on the momentum-only implementation that crashed"
    )


def test_step_counters_are_left_alone() -> None:
    """A scalar is not shape-coupled — padding it would corrupt bias correction."""
    ck = _tiny_ckpt(mults=(1.25,))
    names = list(ck["opt_param_names"])
    i = names.index("blocks.0.ffn.0.bias")
    ck, _ = widen_checkpoint(ck, align=8, seed=3)
    assert ck["opt"]["state"][i]["step"].item() == pytest.approx(1234.0)


def test_a_checkpoint_with_swa_is_REFUSED() -> None:
    """Trainer.load swallows the SWA shape error and keeps RANDOM weights."""
    ck = _tiny_ckpt(mults=(1.25,))
    ck["swa_model"] = {"module.blocks.0.ffn.0.weight": torch.zeros(10, 8)}
    with pytest.raises(ValueError, match="swa_model"):
        widen_checkpoint(ck, align=8, seed=3)


def test_optimizer_state_without_param_names_is_REFUSED() -> None:
    """Silently skipping the migration is the failure this repo keeps shipping.

    ⚑ The match string pins THIS guard, not merely "some ValueError mentioning
    opt_param_names". Mutation-measured: `match="opt_param_names"` let the
    disarmed-guard mutant SURVIVE, because the F5 orphan backstop below raises
    on the same input and its message also contains the phrase. Two guards, one
    regex that cannot tell them apart — a passing test that proved neither.
    """
    ck = _tiny_ckpt(mults=(1.25,))
    ck["opt_param_names"] = []
    with pytest.raises(ValueError, match="cannot be addressed by name"):
        widen_checkpoint(ck, align=8, seed=3)


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


# ---------------------------------------------------------------------------
# Perimeter: every one of these was "accepted, then silently ignored".
# ---------------------------------------------------------------------------

def test_new_ffn0_bias_entries_are_exactly_zero() -> None:
    """⚑ Mutation survivor: nothing pinned the new bias.

    Not function-affecting — ffn.2's zero column kills the contribution either
    way — but it sets the initial activation, hence dL/dW2[:,j], which is the
    mechanism the module docstring's gradient argument rests on. An unpinned
    premise in a load-bearing argument is worth a test.
    """
    ck = _tiny_ckpt(mults=(1.25,))
    old_h = int(8 * 1.25)
    ck, _ = widen_checkpoint(ck, align=8, seed=3)
    new_bias = ck["model"]["blocks.0.ffn.0.bias"][old_h:]
    assert new_bias.numel() > 0
    assert torch.all(new_bias == 0), "new units must start unbiased"


def test_a_nested_chained_optimizer_is_REFUSED() -> None:
    """⚑⚑ THE guard-that-could-not-fire.

    `_ChainedOptimizer.state_dict()` returns {"optimizers": [...]} with NO
    "state" key. A refusal that read opt["state"] saw {}, concluded "nothing to
    migrate", and skipped everything silently — the signature defect, inside the
    guard written to prevent it. Reachable with optimizer: soap.
    """
    ck = _tiny_ckpt(mults=(1.25,))
    ck["opt"] = {"optimizers": [{"state": {0: {"momentum_buffer": torch.zeros(10, 8)}}}]}
    with pytest.raises(ValueError, match="optimizers"):
        widen_checkpoint(ck, align=8, seed=3)


def test_soda_anchors_are_REFUSED() -> None:
    """Full param-shaped clones, siblings of `state`, outside the reset net."""
    ck = _tiny_ckpt(mults=(1.25,))
    ck["opt"]["soda_anchors"] = {0: torch.zeros(10, 8)}
    with pytest.raises(ValueError, match="soda_anchors"):
        widen_checkpoint(ck, align=8, seed=3)


def test_a_wrapped_model_key_still_migrates_its_optimizer_state() -> None:
    """⚑ The regex tolerates `_orig_mod.`; the name lookup did not.

    Result would be a widened parameter beside an un-widened moment. NOTE the
    reachability is DEFENSIVE, not live: `Trainer.save` runs
    `strip_compile_prefix` over `state["model"]`, so a genuine checkpoint has
    bare `blocks.N.ffn.*` keys and `_unwrap` never fires on it. This pins the
    handling for hand-assembled or externally produced state dicts; do not read
    it as "compiled checkpoints were broken".
    """
    ck = _tiny_ckpt(mults=(1.25,))
    ck["model"] = {f"_orig_mod.{k}": v for k, v in ck["model"].items()}
    old_h = int(8 * 1.25)
    ck, changes = widen_checkpoint(ck, align=8, seed=3)
    assert changes, "prefixed keys must still be recognised as FFN tensors"
    names = ck["opt_param_names"]
    i = names.index("blocks.0.ffn.0.weight")
    buf = ck["opt"]["state"][i]["momentum_buffer"]
    param = ck["model"]["_orig_mod.blocks.0.ffn.0.weight"]
    assert buf.shape == param.shape, (
        f"moment {tuple(buf.shape)} != param {tuple(param.shape)} — the wrapper "
        "prefix defeated the by-name lookup"
    )
    assert torch.all(buf[old_h:] == 0)


def test_a_manifest_that_does_not_describe_this_checkpoint_is_REFUSED() -> None:
    """A PRESENT but wrong opt_param_names skipped the migration silently."""
    ck = _tiny_ckpt(mults=(1.25,))
    ck["opt_param_names"] = ["something.else.weight"] * len(ck["opt_param_names"])
    with pytest.raises(ValueError, match="no matching 'opt_param_names'"):
        widen_checkpoint(ck, align=8, seed=3)


def test_variable_width_arch_is_REFUSED() -> None:
    """TransformerBlock sizes the FFN per layer; plan_widths uses the scalar."""
    ck = _tiny_ckpt(mults=(1.25,))
    ck["arch"]["embed_dim_by_layer"] = [8, 16]
    with pytest.raises(ValueError, match="embed_dim_by_layer"):
        widen_checkpoint(ck, align=8, seed=3)


def test_the_shrink_guards_actually_fire() -> None:
    """⚑ Mutation survivor M11: both guards were unpinned.

    They are the only symptom if `arch` and the tensors ever disagree in the
    wrong direction — the tool plans widths from `arch` and pads tensors.
    """
    with pytest.raises(ValueError, match="refusing to SHRINK dim0"):
        _pad_rows(torch.zeros(16, 4), 8, fill=None)
    with pytest.raises(ValueError, match="refusing to SHRINK dim1"):
        _pad_cols(torch.zeros(4, 16), 8)


def test_the_scalar_ffn_mult_is_the_FIRST_layers_mult_not_the_mean() -> None:
    """⚑ R5: the two FFN tools DISAGREE here, and that is fine.

    `scripts/shrink_ffn_checkpoint.py` writes `ffn_mult =
    mean(normalized_schedule)`; this tool writes `new_mults[0]` (1.734375 vs 1.5
    on the production schedule). The comment in widen_ffn_aligned.py used to
    justify the line by claiming shrink "already does" the same thing — false.
    The disagreement is harmless because the scalar is DECORATIVE whenever
    `ffn_mult_by_layer` is present: every in-tree consumer prefers the schedule.
    This test pins which of the two conventions THIS tool follows, so a future
    "reconciliation" has to be a deliberate decision rather than a drift.
    """
    ck = _tiny_ckpt(mults=(1.25, 2.75))  # widths 10, 22 -> aligned 16, 24
    ck, _ = widen_checkpoint(ck, align=8, seed=3)
    schedule = ck["arch"]["ffn_mult_by_layer"]
    assert [int(8 * m) for m in schedule] == [16, 24]
    mean = sum(schedule) / len(schedule)
    assert ck["arch"]["ffn_mult"] == pytest.approx(schedule[0])
    assert ck["arch"]["ffn_mult"] != pytest.approx(mean), (
        "the fixture no longer distinguishes first-mult from mean-mult, so this "
        "test would pass under shrink_ffn_checkpoint.py's convention too"
    )


def test_main_does_not_crash_on_an_already_aligned_scalar_arch(tmp_path) -> None:
    """The documented 'safe to re-run' path raised list(None) and wrote nothing."""
    import scripts.widen_ffn_aligned as mod

    ck = _tiny_ckpt(mults=(2.0,))
    del ck["arch"]["ffn_mult_by_layer"]
    src, dst = tmp_path / "in.pt", tmp_path / "out.pt"
    torch.save(ck, src)
    assert mod.main(["--in", str(src), "--out", str(dst), "--align", "8"]) == 0
    assert dst.exists(), "the no-op path must still write its output"


# ---------------------------------------------------------------------------
# Independent review R1-R7.
# ---------------------------------------------------------------------------

def test_optimizer_state_with_no_state_key_and_no_names_is_REFUSED() -> None:
    """⚑ R1: the F1 fix (`if opt` not `if opt_state`) was UNPINNED.

    `test_optimizer_state_without_param_names_is_REFUSED` supplies an empty
    `opt_param_names` alongside a POPULATED `opt["state"]`, so `opt` and
    `opt["state"]` are both truthy and BOTH spellings of the guard raise —
    reverting the fix left the whole suite green. This is the input that
    separates them: an optimizer state dict with no "state" key at all, which is
    exactly what `_ChainedOptimizer.state_dict()` returns. Under the
    `opt_state`-keyed spelling the guard sees {} and waves the checkpoint
    through with the migration silently skipped.
    """
    ck = _tiny_ckpt(mults=(1.25,))
    ck["opt"] = {"param_groups": [{"params": [0]}]}  # truthy, but no "state"
    del ck["opt_param_names"]
    with pytest.raises(ValueError, match="cannot be addressed by name"):
        widen_checkpoint(ck, align=8, seed=3)


def test_aurora_scale_factor_ratio_is_pinned() -> None:
    """⚑⚑ R2: widening is function-preserving but NOT dynamics-preserving.

    `_aurora_update` ends with `out *= sqrt(max(1, orig_rows / orig_cols))`, so
    growing `ffn.0.weight` from (796, 512) to (896, 512) multiplies the WHOLE
    update — untouched rows included — by sqrt(896/796) = 1.0610. A widened arm
    therefore differs from its control by capacity AND by a per-layer effective
    step-size increase; see the module docstring's confound section, which this
    test exists to keep honest.

    The factor is imported from the real optimizer rather than re-derived: the
    polar factor is stubbed out with an identity so the only thing left in the
    output/input norm ratio is that trailing scale. `pp_iterations=1` skips the
    row-norm rebalance, and unit-norm rows make the internal `scale` exactly 1.
    """
    from chess_anti_engine.train.aurora import _aurora_update

    def scale_factor(rows: int, cols: int) -> float:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(rows, cols, generator=g)
        tall = x if rows >= cols else x.T
        tall = tall / tall.norm(dim=-1, keepdim=True)
        y = tall if rows >= cols else tall.T
        out = _aurora_update(
            y,
            pp_iterations=1,
            polar_method="polar_express",
            polar_express_fn=lambda mat, **_: mat,
        )
        return float(out.norm() / y.norm())

    f_old = scale_factor(796, 512)
    f_new = scale_factor(896, 512)
    assert f_old == pytest.approx(math.sqrt(796 / 512), rel=1e-5)
    assert f_new == pytest.approx(math.sqrt(896 / 512), rel=1e-5)
    assert f_new / f_old == pytest.approx(math.sqrt(896 / 796), rel=1e-5)
    assert f_new / f_old == pytest.approx(1.0610, abs=5e-4), (
        "the +6.1% figure quoted in the module docstring is stale"
    )
    # The output projection (512, h) is UNAFFECTED: max(1.0, 512/h) == 1.0.
    assert scale_factor(512, 896) == pytest.approx(1.0, rel=1e-5)


def test_arch_width_that_disagrees_with_the_stored_tensor_is_REFUSED() -> None:
    """⚑ R3: `changes` came from `arch` and was never checked against `model`."""
    ck = _tiny_ckpt(mults=(1.25,))  # arch says width 10
    ck["model"]["blocks.0.ffn.0.weight"] = torch.zeros(12, 8)  # tensors say 12
    with pytest.raises(ValueError, match="arch width disagrees with the stored tensor"):
        widen_checkpoint(ck, align=8, seed=3)


def test_a_planned_layer_that_widens_nothing_is_REFUSED() -> None:
    """⚑ R3: arch rewritten while zero tensors moved = a checkpoint that lies.

    Model keys the FFN patterns do not match (`layers.N.ffn.*` instead of
    `blocks.N.ffn.*`) used to produce changes={0: (10, 16)}, zero widened
    tensors, and an `arch` claiming width 16. On resume,
    `load_state_dict_tolerant` — called WITHOUT `require_complete=True` — drops
    the mismatched FFN tensors, leaving fresh-init FFNs under the catastrophic
    load threshold, so nothing fires.
    """
    ck = _tiny_ckpt(mults=(1.25,))
    ck["model"] = {k.replace("blocks.", "layers.", 1): v for k, v in ck["model"].items()}
    with pytest.raises(ValueError, match="contributed no tensors to the widening"):
        widen_checkpoint(ck, align=8, seed=3)


def test_variable_width_arch_is_REFUSED_even_when_the_scalar_plan_is_ALIGNED() -> None:
    """⚑ R4: the refusal sat AFTER `if not changes: return`, so it was skippable.

    embed_dim 8 x mult 2.0 = 16, a multiple of 8, so plan_widths finds nothing to
    do and the old code returned "already tile-aligned" — while the width
    TransformerBlock actually builds from `embed_dim_by_layer` is
    int(10 * 2.0) = 20, which is NOT aligned. A clean bill of health for a net
    this tool cannot read.
    """
    ck = _tiny_ckpt(mults=(2.0,))
    ck["arch"]["embed_dim_by_layer"] = [10]
    changes, _ = plan_widths(ck["arch"], align=8)
    assert changes == {}, "the fixture must exercise the ALIGNED (early-return) path"
    with pytest.raises(ValueError, match="embed_dim_by_layer"):
        widen_checkpoint(ck, align=8, seed=3)


def test_swa_is_REFUSED_even_when_the_scalar_plan_is_ALIGNED() -> None:
    """⚑ R4: same early-return skip, for the SWA refusal."""
    ck = _tiny_ckpt(mults=(2.0,))
    ck["swa_model"] = {"module.blocks.0.ffn.0.weight": torch.zeros(16, 8)}
    changes, _ = plan_widths(ck["arch"], align=8)
    assert changes == {}, "the fixture must exercise the ALIGNED (early-return) path"
    with pytest.raises(ValueError, match="swa_model"):
        widen_checkpoint(ck, align=8, seed=3)


def test_the_trainers_real_rekeying_helper_is_what_the_unwrap_docstring_says() -> None:
    """⚑ R6: `_unwrap`'s docstring used to claim parity with a helper that
    does something else — and under a name (`Trainer._wrap_agnostic_name`) that
    does not exist in this repo at all. The real one is the module-level
    `strip_compile_prefix`. This pins the two properties the corrected docstring
    asserts about it, so the correction cannot go stale silently.
    """
    from chess_anti_engine.train.trainer import strip_compile_prefix

    # (1) first occurrence ANYWHERE, not just leading — this is why it reaches
    #     AveragedModel's nested keys.
    got = strip_compile_prefix({"module._orig_mod.blocks.0.ffn.0.weight": 1})
    assert list(got) == ["module.blocks.0.ffn.0.weight"]
    # (2) it never strips a `module.` segment.
    assert list(strip_compile_prefix({"module.blocks.0.ffn.0.weight": 1})) == [
        "module.blocks.0.ffn.0.weight"
    ]


def test_the_printed_schedule_is_valid_yaml_that_round_trips(tmp_path, capsys) -> None:
    """⚑ R7: a Python list repr has to be hand-translated into the launch yaml,
    and the launch yaml is the file that decides whether the widening is loaded
    at all (`_maybe_load_bootstrap` never reads the checkpoint's arch). Print
    the line to paste, and prove it parses back to the same schedule.
    """
    import yaml

    import scripts.widen_ffn_aligned as mod

    ck = _tiny_ckpt(mults=(1.25, 2.75))
    src, dst = tmp_path / "in.pt", tmp_path / "out.pt"
    torch.save(ck, src)
    assert mod.main(["--in", str(src), "--out", str(dst), "--align", "8"]) == 0

    lines = [
        ln for ln in capsys.readouterr().out.splitlines()
        if ln.startswith("ffn_mult_by_layer:")
    ]
    assert len(lines) == 1, f"expected exactly one yaml line, got {lines}"
    parsed = yaml.safe_load(lines[0])
    assert isinstance(parsed, dict), f"{lines[0]!r} is not a yaml mapping"
    emitted = torch.load(dst, map_location="cpu", weights_only=False)
    assert parsed["ffn_mult_by_layer"] == [
        float(m) for m in emitted["arch"]["ffn_mult_by_layer"]
    ]
    assert [int(8 * m) for m in parsed["ffn_mult_by_layer"]] == [16, 24]


def test_format_ffn_mult_yaml_round_trips_the_production_schedule() -> None:
    import yaml

    from scripts.widen_ffn_aligned import format_ffn_mult_yaml

    cur = (
        1.5, 1.5, 1.5, 1.5, 1.5, 1.555556, 1.65, 1.772222,
        1.894444, 1.666667, 1.638889, 1.905556, 1.783333, 1.794444, 1.744444, 1.5,
    )
    new = aligned_ffn_mults(cur, embed_dim=512, align=128)
    line = format_ffn_mult_yaml(new)
    assert yaml.safe_load(line)["ffn_mult_by_layer"] == list(new)


def test_module_docstring_records_the_optimizer_dynamics_CONFOUND() -> None:
    """⚑⚑ R2 is a design confound, not a bug: it must reach the ledger prereg.

    The docstring is the only place this is written down, so pin the load-bearing
    phrases rather than trusting them to survive an edit.
    """
    import scripts.widen_ffn_aligned as mod

    doc = mod.__doc__ or ""
    assert "sqrt(h_new / h_old)" in doc
    assert re.search(r"confound", doc, re.I), "the confound must be named as such"
    assert re.search(r"Confounds", doc), "the ledger prereg line must be named"
    assert re.search(r"_maybe_load_bootstrap", doc), (
        "the deployment path that silently drops the widening must be named"
    )
